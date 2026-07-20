import ast
import torch
import pandas as pd
from torch import nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def default_transform(image_size: int = 256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


class VQVAET5DATASET(Dataset):
    def __init__(self, csv_dir="../dataset/dataset.csv", image_size=256, transform=None):
        self.csv_dir = csv_dir
        self.csv = pd.read_csv(self.csv_dir)
        self.transform = transform or default_transform(image_size)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        img_name = str(self.csv.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        image = self.transform(image)

        labels = self.csv.iloc[idx, 1]

        tokenized_labels = ast.literal_eval(str(self.csv.iloc[idx, 2]))
        tokenized_labels = torch.tensor(tokenized_labels, dtype=torch.long)

        return image, labels, tokenized_labels


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def adopt_weight(disc_factor, i, threshold, value=0.):
    if i < threshold:
        disc_factor = value
    return disc_factor


# if __name__ == "__main__":
#     dataset = VQVAET5DATASET()
#     print(f"Dataset length: {len(dataset)}")