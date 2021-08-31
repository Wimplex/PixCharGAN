import torch
import torchvision.transforms as T
from torchvision.transforms.transforms import ColorJitter


def center(img, max_size=32):
    out = torch.zeros(size=[img.shape[0], max_size, max_size], dtype=img.dtype)
    x_offset = (max_size - img.shape[1]) // 2
    y_offset = (max_size - img.shape[2]) // 2
    out[:, x_offset:x_offset + img.shape[1], y_offset:y_offset + img.shape[2]] = img
    return out


forward_transformation = T.Compose([
    T.ToTensor(),
    T.Lambda(center)
])

augment_transformation = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply(transforms=[T.ColorJitter()], p=0.5),
    forward_transformation
])

reverse_transformation = T.Compose([
    # --> Normalize logic here <--
    T.ToPILImage(mode='RGB')
])