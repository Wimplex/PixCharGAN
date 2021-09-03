import warnings
warnings.filterwarnings('ignore')

import random
import json
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from dataloader import Sprite16x16Dataset
from networks.dcgan import DCGAN_Generator, DCGAN_Discriminator, weights_init_dcgan
from plotting import plot_anim_fixed_noise


REAL_LABEL = 1
FAKE_LABEL = 0


def noise_mix(img_data, p=0.5):
    if np.random.uniform() < p:
        noise = torch.normal(0.0, 0.001, size=img_data.shape, device=img_data.device, requires_grad=False)
        img_data = img_data + noise
    return img_data


def train_one_epoch(generator: torch.nn.Module, discriminator: torch.nn.Module, \
    train_loader: DataLoader, gen_optimizer: optim.Optimizer, disc_optimizer: optim.Optimizer, \
    criterion: nn.Module, device: str, hidden_size: int, fixed_data: dict = None):
    
    imgs_list = []

    for i, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):

        tensor_batch, direction_batch, ouline_batch = data

        # Discriminator training. The data entirely is real (accumulate gradients)
        discriminator.zero_grad()
        real_cpu = tensor_batch.to(device)
        direction_batch = F.one_hot(torch.tensor(direction_batch, device=device), num_classes=4).to(device).float()

        # Add noise in random cases
        real_cpu = noise_mix(real_cpu, p=0.3)

        batch_size = real_cpu.size(0)
        label = torch.full([batch_size,], REAL_LABEL, dtype=torch.float, device=device, requires_grad=True)
        output = discriminator(real_cpu, direction_batch).view(-1)
        loss_D_real = criterion(output, label.detach())
        loss_D_real.backward()

        # Discriminator training. The data entirely is fake (accumulate gradients).
        # Replace few values with direction and outline features
        noise = torch.randn(batch_size, hidden_size, device=device)
        fake = generator(noise, direction_batch)
        fake = noise_mix(fake, p=0.3)

        label = torch.full([batch_size,], FAKE_LABEL, dtype=torch.float, device=device, requires_grad=True)
        output = discriminator(fake.detach(), direction_batch).view(-1)
        loss_D_fake = criterion(output, label.detach())
        loss_D_fake.backward()

        # Update discriminator
        loss_D = loss_D_real + loss_D_fake
        disc_optimizer.step()

        # Generator training. Accumulate gradients for generator
        generator.zero_grad()
        label = torch.full([batch_size,], REAL_LABEL, dtype=torch.float, device=device, requires_grad=True)
        output = discriminator(fake, direction_batch).view(-1)
        loss_G = criterion(output, label.detach())
        loss_G.backward()

        # Update generator
        gen_optimizer.step()


def train(generator: nn.Module, discriminator: nn.Module, train_loader: DataLoader, \
    num_epochs: int, gen_optimizer: optim.Optimizer, disc_optimizer: optim.Optimizer, \
    criterion: nn.Module, device: str, hidden_size: int):

    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    # Generate fixed random noise and conditions
    fixed_noise = torch.randn(64, hidden_size, device=device)
    fixed_dimensions = F.one_hot(torch.randint(1, 4, size=[64,]), num_classes=4).to(device).float()
    fixed_data = {'noise': fixed_noise, 'dimensions': fixed_dimensions}

    imgs_list = []
    for i in range(num_epochs):
        print(f"Epoch {i + 1} started.")
        train_one_epoch(
            generator,
            discriminator,
            train_loader,
            gen_optimizer,
            disc_optimizer,
            criterion,
            device,
            hidden_size,
            fixed_data
        )
        
        # Plot and save visualization
        with torch.no_grad():
            noise = fixed_data['noise']
            dimensions = fixed_data['dimensions']
            fake = generator(noise, dimensions).detach().cpu()
            img_grid = vutils.make_grid(fake, padding=2, normalize=True)
            imgs_list.append(img_grid)
        plot_anim_fixed_noise(imgs_list, f'python/img/{i+1}.png')            


def main(args):
    # Setup randomness
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    # Setup params
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Prepare dataloader
    dataset = Sprite16x16Dataset(args['data_root'], aug_factor=3)
    data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_data_workers'])

    # Instantiate and setup generator
    netG = DCGAN_Generator(hidden_size=args['hidden_size'], n_feature_maps=128)
    netG.apply(weights_init_dcgan)

    # Instantiate and setup discriminator
    netD = DCGAN_Discriminator()
    netD.apply(weights_init_dcgan)

    # Loss
    loss = nn.BCELoss()

    # Optimizers
    optG = optim.Adam(netG.parameters(), lr=args['generator_lr'], betas=[args['beta1'], 0.999])
    optD = optim.Adam(netD.parameters(), lr=args['discriminator_lr'], betas=[args['beta1'], 0.999])

    # Start training process
    train(
        generator=netG,
        discriminator=netD,
        num_epochs=args['num_epochs'],
        gen_optimizer=optG,
        disc_optimizer=optD,
        train_loader=data_loader,
        criterion=loss,
        device=device,
        hidden_size=args['hidden_size']
    )


if __name__ == '__main__':
    args = json.load(open('python/params/param_set1.json', 'rb'))
    main(args)