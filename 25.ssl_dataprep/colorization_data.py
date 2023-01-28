import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import transforms

import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt


class ColorizationDataset(Dataset):
    """
    Implements the SSL Dataset setup for Colourful Image Colorization paper.
    """

    def __init__(self, root):
        self.imgs = glob(root + "/*.png")
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))]
        )

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        inp_img = img.convert('L')
        return img, inp_img

    def __len__(self):
        return len(self.imgs)

