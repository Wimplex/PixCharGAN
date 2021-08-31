import os
import glob
from torch.utils.data import Dataset
from PIL import Image


class Sprite16x16Dataset(Dataset):
    def __init__(self, dataset_root_dir, transforms):
        super(Sprite16x16Dataset, self).__init__()

        sprites_paths_template = os.path.join(dataset_root_dir, '16x16', '*', '*.png')
        self.paths = glob.glob(sprites_paths_template)
        self.transforms = transforms

    def __getitem__(self, idx):
        # --> Some code applying transformations here <---
        img = Image.open(self.paths[idx]).convert('RGB')
        tensor = self.transforms(img)
        return tensor

    def __len__(self):
        return len(self.paths)
