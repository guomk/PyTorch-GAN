import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        # if image_A.mode != "RGB":
        #     image_A = to_rgb(image_A)
        # if image_B.mode != "RGB":
        #     image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))




class OneDomainImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="test", domain="B"):
        self.transform = transforms.Compose(transforms_)
        self.domain = domain

        self.img_dir = os.path.join(root, f"{mode}/imgs")
        # print(self.img_dir)
        self.files = sorted(glob.glob(self.img_dir + "/*.*"))
        # print(len(self.files))

    def __getitem__(self, index):
        image = np.array(Image.open(self.files[index % len(self.files)]).convert("L"))
        # image = np.expand_dims(image, axis=0)

        # image = image.transpose((2, 0, 1))
        # print(image.shape)

        # Convert grayscale images to rgb
        # if image_A.mode != "RGB":
        #     image_A = to_rgb(image_A)
        # if image_B.mode != "RGB":
        #     image_B = to_rgb(image_B)

        item = self.transform(image)
        return {self.domain: item}

    def __len__(self):
        return len(self.files)
