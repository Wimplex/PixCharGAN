import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

# from google.colab import files


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


# def google_colab_zip_and_dwnload(dir_path, output_path):
#    shutil.make_archive(output_path, 'zip', dir_path)
#     files.download(output_path)


def int_to_grayscale_hex(value):
    return '#%02x%02x%02x' % (value, value, value)



def save_checkpoint(modelD: nn.Module, modelG: nn.Module, \
        optimD: optim.Optimizer, optimG: optim.Optimizer, \
        save_path: str):
    state_dict = {
        'discriminator': modelD.state_dict(),
        'generator': modelG.state_dict(),
        'discriminator_optim': optimD.state_dict(),
        'generator_optim': optimG.state_dict()
    }
    torch.save(state_dict, save_path)