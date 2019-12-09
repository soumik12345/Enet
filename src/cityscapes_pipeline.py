import torch
import numpy as np
from glob import glob
from PIL import Image
from torch import LongTensor
from random import random, randint
from config import CITYSCAPES_CONFIG
from torch.utils.data import Dataset
from torch.nn import ConstantPad2d, ZeroPad2d
from torchvision.transforms import Compose, ToTensor, Normalize


def preprocess(image, mask):
    # Apply Flip
    if random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    # Image Transforms
    _transforms = Compose([
        ToTensor(),
        Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    image = _transforms(image)
    # Transform mask to Tensor
    mask = LongTensor(np.array(mask).astype(np.int64))
    # Apply Crop
    crop = CITYSCAPES_CONFIG['crop']
    if crop:
        h, w = image.shape[1], image.shape[2]
        pad_vertical = max(0, crop[0] - h)
        pad_horizontal = max(0, crop[1] - w)
        image = ZeroPad2d((
            0, pad_horizontal,
            0, pad_vertical
        ))(image)
        mask = ConstantPad2d((
                0, pad_horizontal,
                0, pad_vertical
            ),
        255)(mask)
        h, w = image.shape[1], image.shape[2]
        i = randint(0, h - crop[0])
        j = randint(0, w - crop[1])
        image = image[:, i : i + crop[0], j : j + crop[1]]
        mask = mask[i : i + crop[0], j : j + crop[1]]
    return image, mask