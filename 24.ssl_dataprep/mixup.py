import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def mixup(img_1 , img_2 , label_1 , label_2 , lam):
    '''
    Mixes two inputs and labels.
    '''
    mixed_img = lam * img_1 + (1-lam) * img_2
    mixed_label = lam * label_1 + (1-lam) * label_2

    return mixed_img , mixed_label