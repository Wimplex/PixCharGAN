import os
import glob
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.transforms as T
from torch.utils.data import Dataset


def center_pad(img_data, max_size, background_color):
    """ Center small image in a square with <max_size> sides """
    # out = torch.ones(size=[img_data.shape[0], max_size, max_size], dtype=img_data.dtype) * background_color / 255
    out = torch.zeros(size=[img_data.shape[0], max_size, max_size], dtype=img_data.dtype)
    # out[:3,:,:] = background_color / 255
    x_offset = (max_size - img_data.shape[1]) // 2
    y_offset = (max_size - img_data.shape[2]) // 2
    out[:, x_offset:x_offset + img_data.shape[1], y_offset:y_offset + img_data.shape[2]] = img_data
    return out


def horisontal_flip_with_confirmation(img_data, p=0.5):
    """ Return randomly horizontal flipped image and a flip confirmation flag """
    if np.random.uniform() < p: return T.RandomHorizontalFlip(p=1.0)(img_data), True
    else: return img_data, False


def noise_mix(img_data, std=0.001, p=0.5):
    """ Randomly add normal noise to image """
    if np.random.uniform() < p:
        noise = torch.normal(0.0, std, size=img_data.shape, device=img_data.device, requires_grad=False)
        img_data = img_data + noise
    return img_data


class Sprite16x16Dataset(Dataset):
    directions_to_num = {'R': 0, 'L': 1, 'U': 2, 'D': 3}
    background_color = 150

    def __init__(self, dataset_root_dir, aug_factor=1, max_pad_size=32):
        super(Sprite16x16Dataset, self).__init__()

        sprites_paths_template = os.path.join(dataset_root_dir, '16x16', '*', '*.png')
        self.paths = glob.glob(sprites_paths_template) * aug_factor
        self.sprite_directions = []
        self.sprite_outlines = []
        self.aug_factor = aug_factor

        self.__prepare_meta_features()

        self.forward_transformation = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: center_pad(x, max_size=max_pad_size, background_color=self.background_color)),
            T.Normalize((0.485, 0.456, 0.406, 0.0), (0.229, 0.224, 0.225, 1.0))
        ])

        self.aug_transformation = T.Compose([
            T.ColorJitter(),
            # T.Lambda(lambda x: noise_mix(x, p=0.1)),
        ])

    def __prepare_meta_features(self):
        for img_path in self.paths:
            meta_comp = os.path.basename(img_path).split('.')[0].split('_')
            self.sprite_directions.append(meta_comp[2])
            self.sprite_outlines.append(int(meta_comp[3]))

    def __getitem__(self, idx):
        # Read data
        img_path = self.paths[idx]
        img = Image.open(img_path).convert('RGBA')
        
        # Apply transformations
        direction = self.sprite_directions[idx]
        img = self.forward_transformation(img)
        # if idx % self.aug_factor != 0:
        img = self.aug_transformation(img)
        img, flipped = horisontal_flip_with_confirmation(img, p=0.3)
        if flipped: direction = 'R' if direction == 'L' else 'L'
        direction = self.directions_to_num[direction]
        direction = F.one_hot(torch.tensor(direction), num_classes=4)

        return img, direction, self.sprite_outlines[idx]

    def __len__(self):
        return len(self.paths)
