import os
import sys
import torch
import random
import argparse
import gdown
from PIL import Image
from transformers import AutoTokenizer
from training.utils import default_transform
from training.vqvaeT5 import VQVAE_T5, get_device
from transformers import T5ForConditionalGeneration
from word_on_background.generator import generate_image_with_text
from training.pretraining.vqvae_model.vqvae import VQVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Generate word-on-background images and run inference on them.")
    parser.add_argument("--count", type=int, default=10, help="Number of images to generate (default: 10)")
    parser.add_argument("--out", type=str, default="./main_image_dir", help="Output directory (default: ./main_image_dir)")
    parser.add_argument("--txt-file", type=str, default="word_on_background/CollinsScrabbleWords(2019).txt", help="Text file to sample words from")
    parser.add_argument("--font-path", type=str, default="word_on_background/fonts/dejavu-sans-mono.bold.ttf", help="Path to the font file")
    parser.add_argument("--font-size", type=int, default=40, help="Font size for the text")
    parser.add_argument("--width", type=int, default=512, help="Width of the generated image")
    parser.add_argument("--height", type=int, default=512, help="Height of the generated image")
    parser.add_argument("--background-color", type=str, default="white", help="Background color of the image")
    parser.add_argument("--text-color", type=str, default="0,0,0", help="Text color in RGB format (e.g. '0,0,0' for black)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for picking holdout words (default: different words each run)")
    parser.add_argument("--train-seed", type=int, default=42, help="Seed the training set was generated with; must match SEED in train_modal.py")
    parser.add_argument("--train-data-amount", type=int, default=15000, help="Amount of training data to generate (default: 15000)")
    parser.add_argument("--eval", type=int, default=None, metavar="N", help="Score accuracy over N holdout words instead of printing individual predictions")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Images per generate() call when scoring (default: 4 because my pc cannot handle any higher :))")
    return parser.parse_args()

def _is_html(path):
    """True if the file is a saved web page rather than a checkpoint.

    Drive serves a virus-scan warning page instead of the bytes for large
    files, and urlretrieve saves that HTML happily. Without this check the
    bad file sticks around forever, since the exists() test below passes and
    torch.load then fails with an unhelpful zip/pickle error.
    """
    with open(path, "rb") as f:
        return f.read(64).lstrip()[:1] == b"<"

## Download the models if they don't exist
def download_models():
    models = [
        ("19Du0wKuHChn772EMEAjR-WR0kRIVWEPW", "training/pretraining/checkpoints/best.pt", "VQVAE"),
        ("1q_Wj6ORuBN3AZN0glBCsqUMlBTZ4mWnK", "training/checkpoints/best.pt", "VQVAET5"),
    ]

    for file_id, path, label in models:
        if os.path.exists(path) and not _is_html(path):
            continue

        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading {label} model to {path}...")
        gdown.download(id=file_id, output=path, quiet=True)

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

def _levenshtein(a, b):
    """Edit distance between two strings, used for character error rate."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ca != cb),  # substitution
            ))
        previous = current
    return previous[-1]


def evaluate(model, tokenizer, args, words):
    """Report exact-match accuracy and character error rate over `words`."""
    device = get_device()
    transform = default_transform(image_size=256)

    total_distance = total_chars = exact = 0
    failures = []

    print(f"\nScoring {len(words)} holdout words (batch size {args.eval_batch_size})...")
    for start in range(0, len(words), args.eval_batch_size):
        chunk = words[start:start + args.eval_batch_size]

        tensors = []
        for offset, word in enumerate(chunk):
            path = generate_one(start + offset + 1, args, word)
            tensors.append(transform(Image.open(path).convert("RGB")))
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            outputs = model.generate(batch, max_length=128, num_beams=5, early_stopping=True)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for word, prediction in zip(chunk, predictions):
            distance = _levenshtein(word, prediction)
            total_distance += distance
            total_chars += len(word)
            if distance == 0:
                exact += 1
            elif len(failures) < 10:
                failures.append((word, prediction, distance))

        print(f"  {min(start + args.eval_batch_size, len(words))}/{len(words)}", end="\r", flush=True)

    # Character error rate is normalised by reference length, so it stays
    # comparable across words of different lengths.
    print(f"\n\n  exact match:          {exact}/{len(words)} ({exact / len(words):.1%})")
    print(f"  character error rate: {total_distance / total_chars:.1%}")
    print(f"  mean edit distance:   {total_distance / len(words):.2f} chars per word")

    if failures:
        print("\n  sample errors:")
        for word, prediction, distance in failures:
            print(f"    target={word!r:20} predicted={prediction!r:20} distance={distance}")


def main():
    args = parse_args()
    download_models()
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

    train_rng = random.Random(args.train_seed)
    train_words = set(w.strip() for w in train_rng.sample(lines, args.train_data_amount))
    holdout = [w for w in lines if w.strip() not in train_words]

    wanted = args.eval if args.eval else args.count
    if wanted > len(holdout):
        print(f"only {len(holdout)} holdout words available, asked for {wanted}", file=sys.stderr)
        sys.exit(1)
    words = [w.strip() for w in random.Random(args.seed).sample(holdout, wanted)]

    if args.eval:
        evaluate(model, tokenizer, args, words)
        return
    
    print(f"\nGenerating {args.count} image(s) and running inference on them (See {args.out} for the images)...")
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