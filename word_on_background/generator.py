import os
import random
import argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def generate_image_with_text(word, width=512, height=512, font_path="fonts/dejavu-sans-mono.bold.ttf", font_size=20, background_color='white', text_color=(0, 0, 0), output_path='result.png'):
    font = ImageFont.truetype(font_path, size=font_size)
    img = Image.new('RGB', (width, height), color=background_color)
    imgDraw = ImageDraw.Draw(img)

    textWidth = imgDraw.textlength(word, font=font)
    textHeight = font_size
    xText = (width - textWidth) / 2
    yText = (height - textHeight) / 2

    imgDraw.text((xText, yText), word, font=font, fill=text_color)
    img.save(output_path)

def sample_text_from_file(file_path, amount=10):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return random.sample(lines, amount)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate an image with text")
    parser.add_argument("--txt-file", type=str, default="CollinsScrabbleWords(2019).txt", help="Text file for words to display on the image")
    parser.add_argument("--width", type=int, default=512, help="Width of the image")
    parser.add_argument("--height", type=int, default=512, help="Height of the image")
    parser.add_argument("--font-path", type=str, default="fonts/dejavu-sans-mono.bold.ttf", help="Path to the font file")
    parser.add_argument("--font-size", type=int, default=20, help="Font size for the text")
    parser.add_argument("--background-color", type=str, default='white', help="Background color of the image")
    parser.add_argument("--text-color", type=str, default='0,0,0', help="Text color in RGB format (e.g., '0,0,0' for black)")
    parser.add_argument("--save-dir", type=str, default='dataset', help="Save dir for the generated images")

    parser.add_argument("--amount", type=int, default=10, help="Amount of random text samples to select from the file")
    return parser.parse_args()

if __name__ == "__main__":

    
    args = parse_args()
    text_color = tuple(map(int, args.text_color.split(',')))
    rows = []

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(f'{args.save_dir}/images', exist_ok=True)

    words = sample_text_from_file(args.txt_file, args.amount)
    for i, word in enumerate(words):
        output_path = f"{args.save_dir}/images/result_{i}.png"
        rows.append({"image_path": output_path, "word": word.strip()})
        generate_image_with_text(word.strip(), width=args.width, height=args.height, font_path=args.font_path, 
                                 font_size=args.font_size, background_color=args.background_color, text_color=text_color, output_path=output_path)
    
    temp = pd.DataFrame(rows, columns=["image_path", "word"])
    temp.to_csv(f"{args.save_dir}/dataset.csv", index=False)