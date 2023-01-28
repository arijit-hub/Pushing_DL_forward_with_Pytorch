import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
from torchvision.transforms import transforms

from itertools import permutations
from glob import glob
from patchify import patchify
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class PuzzleDataset(Dataset):
    """
    Sets up the dataset instance for puzzle based SSL.
    """

    def __init__(self, root):
        self.imgs = glob(root + "/*.jpg")
        self.combinations = list(permutations(range(0, 9)))
        self.crop_shape = (225, 225)
        self.patch_shape = (75, 75, 3)
        self.final_patch_shape = (3, 64, 64)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).resize((256, 256))
        img = transforms.RandomCrop(self.crop_shape)(img)
        img = np.array(img)
        img_patches = patchify(img, (self.patch_shape), 75)
        img_patch_tensor = torch.zeros(9, *self.final_patch_shape)

        img_num = 0
        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                img_patch = transforms.ToTensor()(img_patches[i, j, 0])
                img_patch_tensor[img_num] = transforms.RandomCrop(
                    (self.final_patch_shape[1], self.final_patch_shape[2])
                )(img_patch)
                img_num += 1

        label = int(random.randint(0, len(self.combinations)))
        combination = self.combinations[label]
        img_patch_tensor = img_patch_tensor[list(combination)]
        return img_patch_tensor, torch.LongTensor([label])  

    def __len__(self):
        return len(self.imgs)


def vis(img, val):
    print(img.shape)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(
        make_grid(img.cpu(), 1).permute(1, 2, 0)
    ) 
    plt.show()