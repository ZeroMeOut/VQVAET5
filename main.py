import os
import sys
import torch
import random
import argparse
from PIL import Image
from torchvision import transforms
from training.utils import default_transform
from receipt_generator.generate_receipts import generate_one
from training.vqvaeT5 import VQVAE_T5, load_models, get_device

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic supermarket receipts (image + JSON).")
    parser.add_argument("--count", type=int, default=20, help="Number of receipts to generate (default: 20)")
    parser.add_argument("--out", type=str, default="./main_image_dir", help="Output directory (default: ./main_image_dir)")
    parser.add_argument("--store", type=str, default="random", choices=["asda", "aldi", "random"],
                            help="Which store style to use (default: random mix)")
    parser.add_argument("--style", type=str, default="photo", choices=["clean", "photo", "random"],
                            help="clean = crisp digital look, photo = photographed look "
                                "(creases/lighting/blur/noise), random = mix of both (default: photo)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--generate_json", type=bool, default=False, help="Whether to generate JSON files for the receipts (default: False)")
    return parser.parse_args()


def load_model() -> "VQVAE_T5":
    ckpt_path = "training/checkpoints/best.pt"
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    model = load_models(checkpoint_path='training/encoder_pretraining/checkpoints/best.pt').to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main():
    args = parse_args()
    # model = load_model()
    # print("Model loaded successfully.")

    if args.count < 1:
        print("--count must be >= 1", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    for i in range(1, args.count + 1):
        path = generate_one(i, args, rng=random.Random(args.seed))
        if args.count <= 50 or i % max(1, args.count // 20) == 0 or i == args.count:
            print(f"[{i}/{args.count}] {path}")


if __name__ == "__main__":
    main()
