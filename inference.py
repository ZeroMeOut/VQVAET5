import os
import sys
import torch
import random
import argparse
from PIL import Image
from transformers import AutoTokenizer
from training.utils import default_transform
from training.vqvaeT5 import VQVAE_T5, get_device
from transformers import T5ForConditionalGeneration
from receipt_generator.generate_receipts import generate_one
from training.encoder_pretraining.vqvae_model.vqvae import VQVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic supermarket receipts (image + JSON).")
    parser.add_argument("--count", type=int, default=1, help="Number of receipts to generate (default: 20)")
    parser.add_argument("--out", type=str, default="./main_image_dir", help="Output directory (default: ./main_image_dir)")
    parser.add_argument("--store", type=str, default="random", choices=["asda", "aldi", "random"],
                            help="Which store style to use (default: random mix)")
    parser.add_argument("--style", type=str, default="photo", choices=["clean", "photo", "random"],
                            help="clean = crisp digital look, photo = photographed look "
                                "(creases/lighting/blur/noise), random = mix of both (default: photo)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--generate_json", type=bool, default=False, help="Whether to generate JSON files for the receipts (default: False)")
    return parser.parse_args()

## A different version of the load_model function for inference. There is probably a better way to do this
def load_models(
    checkpoint_path: str = "training/checkpoints/best.pt",
    vqvae_checkpoint_path: str = "training/encoder_pretraining/checkpoints/best.pt",
    t5_name: str = "t5-small",
    freeze_vqvae: bool = False,
) -> "VQVAE_T5":
    device = get_device()

    vqvae_ckpt = torch.load(vqvae_checkpoint_path, map_location=device)
    ckpt = torch.load(checkpoint_path, map_location=device)

    vqvae_model = VQVAE(
        h_dim=vqvae_ckpt["args"]["h_dim"],
        res_h_dim=vqvae_ckpt["args"]["res_h_dim"],
        n_res_layers=vqvae_ckpt["args"]["n_res_layers"],
        n_embeddings=vqvae_ckpt["args"]["n_embeddings"],
        embedding_dim=vqvae_ckpt["args"]["embedding_dim"],
        beta=vqvae_ckpt["args"]["beta"],
    ).to(device)

    t5_model = T5ForConditionalGeneration.from_pretrained(t5_name)

    model = VQVAE_T5(vqvae_model, t5_model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if freeze_vqvae:
        for param in model.vqvae_model.parameters():
            param.requires_grad = False

    return model


def main():
    args = parse_args()
    model = load_models()
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    if args.count < 1:
        print("--count must be >= 1", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    for i in range(1, args.count + 1):
        path = generate_one(i, args, rng=random.Random(args.seed))

        image = Image.open(path).convert("RGB")
        transform = default_transform(image_size=256)
        image_tensor = transform(image).unsqueeze(0).to(get_device())

        with torch.no_grad():
            output = model.generate(image_tensor, max_length=512, num_beams=5, early_stopping=True)
            print(tokenizer.decode(output[0], skip_special_tokens=True))

        # if args.count <= 50 or i % max(1, args.count // 20) == 0 or i == args.count:
        #     print(f"[{i}/{args.count}] {path}")


if __name__ == "__main__":
    main()
