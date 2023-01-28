import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import transforms
from torchvision.transforms.functional import rotate
from torchvision.utils import make_grid , save_image

import random
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt


class RotNetDataset(Dataset):
    """
    Adopts the RotNet data preparation.
    """

    def __init__(self, root, transform=None):
        self.transform = transform
        self.imgs = glob(root + "/*.jpg")

    def _img_prep(self, img, label):
        rotation = int(label.item() * 90)
        img = rotate(img, rotation)
        aug = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])

        return aug(img).unsqueeze(0)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        label_0 = torch.LongTensor([0])
        label_1 = torch.LongTensor([1])
        label_2 = torch.LongTensor([2])
        label_3 = torch.LongTensor([3])
        img_0_tensor = self._img_prep(img, label_0)
        img_1_tensor = self._img_prep(img, label_1)
        img_2_tensor = self._img_prep(img, label_2)
        img_3_tensor = self._img_prep(img, label_3)

        labels = torch.cat((label_0, label_1, label_2, label_3), dim=0)
        imgs = torch.cat(
            (img_0_tensor, img_1_tensor, img_2_tensor, img_3_tensor), dim=0
        )

        return imgs, labels

    def __len__(self):
        return len(self.imgs)


def custom_collate(batch):
    batch_imgs, batch_labels = default_collate(batch)
    batch_imgs = batch_imgs.reshape(
        batch_imgs.shape[0] * batch_imgs.shape[1],
        batch_imgs.shape[2],
        batch_imgs.shape[3],
        batch_imgs.shape[4],
    )
    return batch_imgs, batch_labels


def vis(img):
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(make_grid(img.detach().cpu(), 4).permute(1, 2, 0))
    plt.show()


