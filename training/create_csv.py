import os
import json
import pandas as pd
from transformers import AutoTokenizer


image_path = "../dataset/images"
json_path = "../dataset/json"

## Honestly this isn't needed because it alsways does the same naming convension butttt just incase.
image_path_list = sorted(os.listdir(image_path))
json_path_list = sorted(os.listdir(json_path))

tokenizer = AutoTokenizer.from_pretrained("t5-small")

## Appending then writing to csv is faster than writing to csv each iteration, so we do that instead.
rows = []

for i, j in zip(image_path_list, json_path_list):
    with open(os.path.join(json_path, j), 'r') as f:
        json_data = json.load(f)

    items = ",".join(k["name"] for k in json_data["items"]) + "."

    tokenized_items = tokenizer(
        items, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )["input_ids"][0]

    tokenized_items = tokenized_items.masked_fill(
        tokenized_items == tokenizer.pad_token_id, -100
        ).tolist()

    rows.append({
        "image_path": os.path.join(image_path, i),
        "items": items,
        "tokenized_items": tokenized_items,
    })

temp = pd.DataFrame(rows, columns=["image_path", "items", "tokenized_items"])
temp.to_csv("../dataset/dataset.csv", index=False)

smaller_temp = pd.DataFrame(rows, columns=["image_path", "items", "tokenized_items"]).sample(n=2000, random_state=42)
smaller_temp.to_csv("../dataset/smaller_dataset.csv", index=False)