import json
import torch


class Config:
    # Labels
    REAL_LABEL = 1
    FAKE_LABEL = 0

    # Common
    SEED = 24
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # NUM_GPUS = 1

    # Project
    DATA_ROOT = 'data/cleaned'
    PROJECT_DIR = 'python'
    OUTPUT_PLOTS_DIR = 'python/img'

    # Data
    IMAGE_SHAPE = [32, 32, 4]
    BATCH_SIZE = 96

    # Training process
    NUM_EPOCHS = 500
    HIDDEN_SIZE = 64
    GENERATOR_LR = 6e-3
    DISCRIMINATOR_LR = 1e-4
    BETA1 = 0.5
    SAVE_CHECKPOINT_EVERY = 30 # <-- could be None if it is not needed to save checkpoints

    def __init__(self):
        pass

    def save_confg(self, save_path: str):
        json.dump(self.__dict__(), open(save_path, 'w'), indent='\t')
    
    def load_config(self, saved_config_path: str):
        params = json.load(open(saved_config_path, 'r'))
        for key, val in params.items():
            setattr(self, key, val)