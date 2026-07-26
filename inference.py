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
from word_on_background.generator import generate_image_with_text
from training.pretraining.vqvae_model.vqvae import VQVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Generate word-on-background images and run inference on them.")
    parser.add_argument("--count", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument("--out", type=str, default="./main_image_dir", help="Output directory (default: ./main_image_dir)")
    parser.add_argument("--txt-file", type=str, default="word_on_background/CollinsScrabbleWords(2019).txt",
                         help="Text file to sample words from")
    parser.add_argument("--font-path", type=str, default="word_on_background/fonts/dejavu-sans-mono.bold.ttf",
                         help="Path to the font file")
    parser.add_argument("--font-size", type=int, default=40, help="Font size for the text")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image")
    parser.add_argument("--background-color", type=str, default="white", help="Background color of the image")
    parser.add_argument("--text-color", type=str, default="0,0,0", help="Text color in RGB format (e.g. '0,0,0' for black)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train-data-amount", type=int, default=15000, help="Amount of training data to generate (default: 15000)")
    return parser.parse_args()

## A different version of the load_model function for inference. There is probably a better way to do this
def load_models(
    checkpoint_path: str = "training/checkpoints/best.pt",
    vqvae_checkpoint_path: str = "training/pretraining/checkpoints/best.pt",
    t5_name: str = "t5-small",
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

    return model


def generate_one(index, args, word):
    images_dir = os.path.join(args.out, "images")
    img_path = os.path.join(images_dir, f"word_{index:05d}.png")
    text_color = tuple(map(int, args.text_color.split(",")))

    generate_image_with_text(
        word,
        width=args.width,
        height=args.height,
        font_path=args.font_path,
        font_size=args.font_size,
        background_color=args.background_color,
        text_color=text_color,
        output_path=img_path,
    )

    return img_path


def main():
    args = parse_args()
    model = load_models()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    if args.count < 1:
        print("--count must be >= 1", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    with open(args.txt_file) as f:
        lines = f.readlines()

    rng = random.Random(args.seed)
    train_words = set(w.strip() for w in rng.sample(lines, args.train_data_amount))
    holdout = [w for w in lines if w.strip() not in train_words]
    words = [w.strip() for w in rng.sample(holdout, args.count)]

    for i, word in enumerate(words, start=1):
        path = generate_one(i, args, word)

        image = Image.open(path).convert("RGB")
        transform = default_transform(image_size=256)
        image_tensor = transform(image).unsqueeze(0).to(get_device())

        with torch.no_grad():
            output = model.generate(image_tensor, max_length=128, num_beams=5, early_stopping=True)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)

        print(f"[{i}/{args.count}] target={word!r} predicted={prediction!r}")


if __name__ == "__main__":
    main()