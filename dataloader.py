import os
import glob
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.transforms as T
from torch.utils.data import Dataset


# Maps direction presented in string format into radians (scaled from 0 to 3*pi/2)
# DIRECTION_TO_RAD = {'R': 0, 'U': 0.333, 'L': 0.666, 'D': 1.0}
DIRECTIONS = {'R': 0, 'L': 1, 'U': 2, 'D': 3}

def center(img, max_size=32):
    """ Center small image in a square with <max_size> sides """
    out = torch.zeros(size=[img.shape[0], max_size, max_size], dtype=img.dtype)
    x_offset = (max_size - img.shape[1]) // 2
    y_offset = (max_size - img.shape[2]) // 2
    out[:, x_offset:x_offset + img.shape[1], y_offset:y_offset + img.shape[2]] = img
    return out


def horisontal_flip_with_confirmation(img, p=0.5):
    """ Return randomly horizontal flipped image and a flip confirmation flag """
    if np.random.uniform() < p:
        return T.RandomHorizontalFlip(p=1.0)(img), True
    else:
        return img, False


class Sprite16x16Dataset(Dataset):
    def __init__(self, dataset_root_dir, aug_factor=1):
        super(Sprite16x16Dataset, self).__init__()

        sprites_paths_template = os.path.join(dataset_root_dir, '16x16', '*', '*.png')
        self.paths = glob.glob(sprites_paths_template) * aug_factor
        self.sprite_directions = []
        self.sprite_outlines = []
        self.aug_factor = aug_factor

        self.__prepare_meta_features()

        self.forward_transformation = T.Compose([
            T.ToTensor(),
            T.Lambda(center),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __prepare_meta_features(self):
        for img_path in self.paths:
            meta_comp = os.path.basename(img_path).split('.')[0].split('_')
            self.sprite_directions.append(meta_comp[2])
            self.sprite_outlines.append(int(meta_comp[3]))

    def __getitem__(self, idx):

        # Read data
        curr_img_path = self.paths[idx]
        img = Image.open(curr_img_path).convert('RGB')
        direction = self.sprite_directions[idx]

        # Apply transformations and augmentations
        if idx % self.aug_factor != 0:
            img = T.ColorJitter()(img)
            img, flipped = horisontal_flip_with_confirmation(img, p=0.3)
            if flipped:
                direction = 'R' if direction == 'L' else 'L'

        direction = DIRECTIONS[direction]
        img = self.forward_transformation(img)

        return img, direction, self.sprite_outlines[idx]

    def __len__(self):
        return len(self.paths)
