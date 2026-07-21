import pandas as pd
from transformers import AutoTokenizer

data = pd.read_csv("../dataset/dataset.csv")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

def tokenize_word(word):
    tokenized_word = tokenizer(
        word, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )["input_ids"][0]

    tokenized_word = tokenized_word.masked_fill(
        tokenized_word == tokenizer.pad_token_id, -100
    ).tolist()

    return tokenized_word

data["tokenized_word"] = data["word"].apply(tokenize_word)

data.to_csv("../dataset/dataset_tokenized.csv", index=False)