## Implements the SIMCLR data preparation ##

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from glob import glob

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class SimCLRDataset(Dataset):
    """
    Implements the SimCLR Dataset
    """

    def __init__(self, data_path, image_size):
        self.data = glob(data_path + "/*.jpg")

        self.aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=image_size, scale=(0.08, 1)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            int(0.10 * image_size[0]), sigma=(0.1, 2.0)
                        )
                    ],
                    p=0.5,
                ),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])

        img_1 = self.aug(img)
        img_2 = self.aug(img)

        return img_1, img_2

    def __len__(self):
        return len(self.data)

