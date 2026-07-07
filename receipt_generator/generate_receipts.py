#!/usr/bin/env python3
"""
Synthetic supermarket receipt generator.

Generates pairs of:
  - receipt_0001.jpg   a photographed-look (or clean) receipt image
  - receipt_0001.json  the exact structured ground truth for that receipt

Usage:
    python3 generate_receipts.py --count 500 --out ./dataset
    python3 generate_receipts.py --count 50 --out ./dataset --style clean
    python3 generate_receipts.py --count 50 --out ./dataset --store aldi
    python3 generate_receipts.py --count 50 --out ./dataset --seed 42
"""

import argparse
import json
import os
import random
import sys
import time

from .receipt_gen import build_receipt, render_receipt, make_photographed


def strip_internal_fields(receipt):
    """Removes the '_style' bookkeeping block so the saved JSON only
    contains the actual receipt content (the real training target)."""
    clean = {k: v for k, v in receipt.items() if not k.startswith("_")}
    return clean


def generate_one(index, args, rng):
    store_key = None if args.store == "random" else args.store
    seed = rng.randint(0, 2**31 - 1)
    receipt = build_receipt(store_key=store_key, seed=seed)

    clean_img = render_receipt(receipt)

    if args.style == "clean":
        final_img = clean_img.convert("RGB")
    elif args.style == "photo":
        final_img = make_photographed(clean_img, rng_seed=seed)
    else:  # random: mix of both, weighted toward photo since that's the OCR-robustness use case
        final_img = make_photographed(clean_img, rng_seed=seed) if rng.random() < 0.85 \
            else clean_img.convert("RGB")

    stem = f"receipt_{index:05d}"
    images_dir = os.path.join(args.out, "images")
    img_path = os.path.join(images_dir, f"{stem}.jpg")
    final_img.save(img_path, format="JPEG", quality=90)

    if args.generate_json:
        json_dir = os.path.join(args.out, "json")
        json_path = os.path.join(json_dir, f"{stem}.json")
        with open(json_path, "w") as f:
            json.dump(strip_internal_fields(receipt), f, indent=2)

    return img_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic supermarket receipts (image + JSON).")
    parser.add_argument("--count", type=int, default=20, help="Number of receipts to generate (default: 20)")
    parser.add_argument("--out", type=str, default="./dataset", help="Output directory (default: ./dataset)")
    parser.add_argument("--store", type=str, default="random", choices=["asda", "aldi", "random"],
                         help="Which store style to use (default: random mix)")
    parser.add_argument("--style", type=str, default="photo", choices=["clean", "photo", "random"],
                         help="clean = crisp digital look, photo = photographed look "
                              "(creases/lighting/blur/noise), random = mix of both (default: photo)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--generate_json", type=bool, default=True, help="Whether to generate JSON files for the receipts (default: True)")
    args = parser.parse_args()

    if args.count < 1:
        print("--count must be >= 1", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "json"), exist_ok=True)
    rng = random.Random(args.seed)

    start = time.time()
    for i in range(1, args.count + 1):
        path = generate_one(i, args, rng)
        if args.count <= 50 or i % max(1, args.count // 20) == 0 or i == args.count:
            print(f"[{i}/{args.count}] {path}")

    elapsed = time.time() - start
    print(f"\nDone. {args.count} receipts written to '{args.out}' in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
