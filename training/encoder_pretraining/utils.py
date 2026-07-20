import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def default_transform(image_size: int = 256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


class VQVAEDATASET(Dataset):
    def __init__(self, root_dir="../dataset/images", image_size=256, transform=None):
        self.root_dir = root_dir
        self.images_list = sorted(os.listdir(self.root_dir))
        self.transform = transform or default_transform(image_size)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images_list[idx])
        image = Image.open(img_name).convert("RGB")
        image = self.transform(image)
        return image