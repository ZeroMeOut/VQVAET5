import torch
import argparse
from training.vqvaeT5 import VQVAE_T5, load_models, get_device
from receipt_generator.generate_receipts import generate_one

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic supermarket receipts (image + JSON).")
    parser.add_argument("--count", type=int, default=1, help="Number of receipts to generate (default: 20)")
    parser.add_argument("--out", type=str, default="./dataset", help="Output directory (default: ./dataset)")
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
    model = load_model()
    print("Model loaded successfully.")


if __name__ == "__main__":
    main()
