import numpy as np
import matplotlib.pyplot as plt


def draw_img(img_grid, save_path):
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(np.transpose(img_grid, [1, 2, 0]))
    plt.savefig(save_path)