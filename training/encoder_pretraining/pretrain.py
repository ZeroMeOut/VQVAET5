import os
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from utils import VQVAEDATASET
from vqvae_model.vqvae import VQVAE

torch.backends.cudnn.benchmark = True  
torch.random.manual_seed(42)


def parse_args():
    p = argparse.ArgumentParser(description="Train the VQVAE receipt encoder")

    p.add_argument("--data-dir", type=str, default="../../dataset/images")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--h-dim", type=int, default=128)
    p.add_argument("--res-h-dim", type=int, default=32)
    p.add_argument("--n-res-layers", type=int, default=2)
    p.add_argument("--n-embeddings", type=int, default=512)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.25)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", type=str, default=None,
                    help="path to a checkpoint .pt file to resume from")

    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_dataloaders(args):
    dataset = VQVAEDATASET(root_dir=args.data_dir, image_size=args.image_size)

    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_recon, total_embed, total_perplexity, n_batches = 0.0, 0.0, 0.0, 0
    original_and_recon_images = []
    for i, x in enumerate(loader):
        x = x.to(device)
        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = F.mse_loss(x_hat, x)
        total_recon += recon_loss.item()
        total_embed += embedding_loss.item()
        total_perplexity += perplexity.item()
        n_batches += 1

        if i == 0: 
            original_and_recon_images.append(x)
            original_and_recon_images.append(x_hat)

    model.train()
    return total_recon / n_batches, total_embed / n_batches, total_perplexity / n_batches, original_and_recon_images


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    train_loader, val_loader = make_dataloaders(args)
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model = VQVAE(
        h_dim=args.h_dim,
        res_h_dim=args.res_h_dim,
        n_res_layers=args.n_res_layers,
        n_embeddings=args.n_embeddings,
        embedding_dim=args.embedding_dim,
        beta=args.beta,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()

        for x in train_loader:
            x = x.to(device)

            optimizer.zero_grad()
            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = F.mse_loss(x_hat, x)
            loss = recon_loss + embedding_loss

            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if global_step % args.log_every == 0:
                print(
                    f"epoch {epoch} step {global_step} "
                    f"loss {loss.item():.4f} recon {recon_loss.item():.4f} "
                    f"embed {embedding_loss.item():.4f} perplexity {perplexity.item():.2f}"
                )
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/recon_loss", recon_loss.item(), global_step)
                writer.add_scalar("train/embedding_loss", embedding_loss.item(), global_step)
                writer.add_scalar("train/perplexity", perplexity.item(), global_step)

            global_step += 1

        val_recon, val_embed, val_perplexity, original_and_recon_images = evaluate(model, val_loader, device)
        val_loss = val_recon + val_embed
        print(
            f"== epoch {epoch} done in {time.time() - epoch_start:.1f}s | "
            f"val_loss {val_loss:.4f} val_recon {val_recon:.4f} "
            f"val_embed {val_embed:.4f} val_perplexity {val_perplexity:.2f} =="
        )
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/recon_loss", val_recon, epoch)
        writer.add_scalar("val/embedding_loss", val_embed, epoch)
        writer.add_scalar("val/perplexity", val_perplexity, epoch)

        if original_and_recon_images:
            images_to_log = torch.cat(original_and_recon_images, dim=0)
            grid = make_grid(images_to_log, nrow=args.batch_size, normalize=True, scale_each=True)
            writer.add_image("val/original_and_reconstructed", grid, epoch)

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, "last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, os.path.join(args.ckpt_dir, "best.pt"))
            print(f"New best model saved (val_loss={val_loss:.4f})")

    writer.close()


if __name__ == "__main__":
    main()