import os
import torch
import argparse
from torch.optim import AdamW
from utils import VQVAET5DATASET
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from vqvaeT5 import VQVAE_T5, load_models, get_device
from torch.optim.lr_scheduler import CosineAnnealingLR

# Reduces fragmentation on small GPUs — set before any CUDA allocations happen
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")



def parse_args():
    p = argparse.ArgumentParser(description="Train the VQVAET5 model")

    p.add_argument("--csv-dir", type=str, default="../dataset/dataset_tokenized.csv")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8) ## Anything higher than 4 here is too much for my laptop
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", dest="num_epochs", type=int, default=100)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding-loss-weight", type=float, default=0.25)
    p.add_argument("--lr-t5", type=float, default=1e-4)
    p.add_argument("--lr-vqvae", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision training (fp16)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=5, help="Save a checkpoint every N epochs")

    p.add_argument("--log-dir", type=str, default="runs")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_loaders(cfg: dict, device: torch.device):
    dataset = VQVAET5DATASET(csv_dir=cfg["csv_dir"], image_size=cfg["image_size"])

    val_size = int(len(dataset) * cfg["val_split"])
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # pin_memory only helps on CUDA; can cause issues on MPS
    pin = device.type == "cuda"
    loader_kwargs = dict(num_workers=cfg["num_workers"], pin_memory=pin)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)

    return train_loader, val_loader


def make_optimizer(model: VQVAE_T5, cfg: dict) -> AdamW:
    return AdamW(
        [
            {"params": model.vqvae_model.parameters(), "lr": cfg["lr_vqvae"]},
            {"params": model.t5_model.parameters(),    "lr": cfg["lr_t5"]},
            {"params": model.image_proj.parameters(),  "lr": cfg["lr_t5"]},
        ],
        weight_decay=cfg["weight_decay"],
    )


def run_epoch(model, loader, device, cfg, optimizer=None, scaler=None):
    """One pass over `loader`. Pass optimizer=None for validation."""
    is_train = optimizer is not None
    use_amp  = cfg["mixed_precision"] and device.type == "cuda"
    model.train() if is_train else model.eval()

    total_loss = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, _, tokenized_labels in loader:
            images           = images.to(device)
            tokenized_labels = tokenized_labels.to(device)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, _, t5_output = model(images, labels=tokenized_labels)
                loss = t5_output.loss

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    optimizer.step()

            total_loss += loss.item()

    n = len(loader)
    return {
        "loss": total_loss / n,
    }


def log(prefix: str, metrics: dict, epoch: int, num_epochs: int):
    print(
        f"Epoch {epoch:>3}/{num_epochs} | {prefix} | "
        f"loss {metrics['loss']:.4f}"
    )


def save_checkpoint(path: str, epoch: int, model, optimizer, scheduler, val_loss: float):
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss":             val_loss,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(cfg: dict):
    torch.manual_seed(cfg["seed"])
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = make_loaders(cfg, device)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = load_models().to(device)

    # Gradient checkpointing trades compute for memory — recomputes activations
    # on the backward pass instead of storing them. Worthwhile on a small GPU.
    model.t5_model.gradient_checkpointing_enable()

    optimizer = make_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])

    use_amp = cfg["mixed_precision"] and device.type == "cuda"
    scaler  = GradScaler(device="cuda") if use_amp else None
    if use_amp:
        print("Mixed precision (fp16) enabled")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    best_val_loss = float("inf")

    writer = SummaryWriter(log_dir=cfg["log_dir"])
    try:
        for epoch in range(1, cfg["num_epochs"] + 1):
            train_metrics = run_epoch(model, train_loader, device, cfg, optimizer=optimizer, scaler=scaler)
            val_metrics   = run_epoch(model, val_loader,   device, cfg, optimizer=None)

            log("train", train_metrics, epoch, cfg["num_epochs"])
            log("val  ", val_metrics,   epoch, cfg["num_epochs"])

            # -- TensorBoard --
            # Group train/val on the same chart by sharing the parent tag

            writer.add_scalars("vqvaet5", {
                "train": train_metrics["loss"],
                "val":   val_metrics["loss"],
            }, epoch)

            # Log the current LR for each param group
            ## writer.add_scalar("LR/vqvae", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("LR/t5",    optimizer.param_groups[1]["lr"], epoch)

            scheduler.step()

            # Save best checkpoint
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    os.path.join(cfg["checkpoint_dir"], "best.pt"),
                    epoch, model, optimizer, scheduler, best_val_loss,
                )
                print(f"  -> New best saved (val loss {best_val_loss:.4f})")

            # Periodic snapshot
            if epoch % cfg["checkpoint_every"] == 0:
                save_checkpoint(
                    os.path.join(cfg["checkpoint_dir"], f"epoch_{epoch:03d}.pt"),
                    epoch, model, optimizer, scheduler, val_metrics["loss"],
                )
    finally:
        writer.close()


if __name__ == "__main__":
    args = parse_args()
    train(vars(args))