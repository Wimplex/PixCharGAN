import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_anim_fixed_noise(data, save_path):
    fig = plt.figure(figsize=(7, 7))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in data]
    ani = animation.ArtistAnimation(fig, ims, interval=100, repeat_delay=1000, blit=True)
    ani.save(save_path)