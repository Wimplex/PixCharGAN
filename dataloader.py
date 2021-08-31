import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from transformations import forward_transformation, augment_transformation


class Sprite16x16Dataset(Dataset):
    def __init__(self, dataset_root_dir, aug_factor=1):
        super(Sprite16x16Dataset, self).__init__()

        sprites_paths_template = os.path.join(dataset_root_dir, '16x16', '*', '*.png')
        self.paths = glob.glob(sprites_paths_template)
        self.aug_factor = aug_factor

    def __getitem__(self, idx):
        # --> Some code applying transformations here <---
        img = Image.open(self.paths[idx]).convert('RGB')
        if idx % self.aug_factor != 0:
            tensor = augment_transformation(img)
        else:
            tensor = forward_transformation(img)
        
        return tensor

    def __len__(self):
        return len(self.paths) * self.aug_factor
