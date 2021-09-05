import warnings
warnings.filterwarnings('ignore')

import os
import random
import json
import tqdm
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import Sprite16x16Dataset, noise_mix
from networks.dcgan import DCGAN_Generator, DCGAN_Discriminator, weights_init_dcgan
from plotting import plot_anim_fixed_noise
from utils import save_checkpoint
from config import REAL_LABEL, FAKE_LABEL, PROJECT_DIR, CHECKPOINTS_DIR, IMAGE_SHAPE


# Instantiate summary writer accessible for every function in a script
train_writer = SummaryWriter()


def train_one_epoch(generator: torch.nn.Module, discriminator: torch.nn.Module, \
    train_loader: DataLoader, gen_optimizer: optim.Optimizer, disc_optimizer: optim.Optimizer, \
    criterion: nn.Module, device: str, hidden_size: int, current_epoch: int, fixed_data: dict = None):
    
    for i, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        curr_iter = current_epoch * len(train_loader) + i
        tensor_batch, direction_batch, outline_batch = data
        direction_batch = F.one_hot(torch.tensor(direction_batch, device=device), num_classes=4)

        # Discriminator training. The data entirely is real (accumulate gradients)
        discriminator.zero_grad()
        real = tensor_batch.to(device)
        batch_size = real.size(0)
        label = torch.full([batch_size,], REAL_LABEL, dtype=torch.float, device=device, requires_grad=True)
        output = discriminator(real).view(-1)
        loss_D_real = criterion(output, label.detach())
        loss_D_real.backward()

        # Discriminator training. The data entirely is fake (accumulate gradients).
        # Replace few values with direction and outline features
        noise = torch.randn(batch_size, hidden_size, device=device)
        fake = generator(noise, direction_batch)
        fake = noise_mix(fake, p=0.2)

        label = torch.full([batch_size,], FAKE_LABEL, dtype=torch.float, device=device, requires_grad=True)
        output = discriminator(fake.detach()).view(-1)
        loss_D_fake = criterion(output, label.detach())
        loss_D_fake.backward()

        # Update discriminator
        loss_D = loss_D_real.item() + loss_D_fake.item()
        disc_optimizer.step()

        # Generator training. Accumulate gradients for generator
        generator.zero_grad()
        label = torch.full([batch_size,], REAL_LABEL, dtype=torch.float, device=device, requires_grad=True)
        output = discriminator(fake).view(-1)
        loss_G = criterion(output, label.detach())
        loss_G.backward()

        # Update generator
        gen_optimizer.step()

        # Log losses
        train_writer.add_scalars('pix_gan_losses', {
            'gen_loss': loss_G.item(),
            'disc_loss': loss_D
        }, curr_iter)


def train(generator: nn.Module, discriminator: nn.Module, train_loader: DataLoader, \
    num_epochs: int, gen_optimizer: optim.Optimizer, disc_optimizer: optim.Optimizer, \
    criterion: nn.Module, device: str, hidden_size: int):

    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    # Generate fixed random noise and conditions
    fixed_noise = torch.randn(64, hidden_size, device=device)
    fixed_dimensions = F.one_hot(torch.randint(0, 4, size=[64,]), num_classes=4).to(device)
    fixed_data = {'noise': fixed_noise, 'dimensions': fixed_dimensions}

    # Setup checkpoints dir
    curr_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
    checkpoints_dir = os.path.join(CHECKPOINTS_DIR, curr_time)
    os.makedirs(checkpoints_dir)

    imgs_list = []
    for i in range(num_epochs):
        print(f"Epoch {i + 1} started.")
        train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            criterion=criterion,
            device=device,
            hidden_size=hidden_size,
            fixed_data=fixed_data,
            current_epoch=i
        )
        
        # Plot and save visualization
        with torch.no_grad():
            noise = fixed_data['noise']
            dimensions = fixed_data['dimensions']
            fake = generator(noise, dimensions).detach().cpu()
            img_grid = vutils.make_grid(fake, padding=2, normalize=True)
            imgs_list.append(img_grid)
        plot_anim_fixed_noise(imgs_list, f'python/img/{i+1}.png')            

        # Save checkpoint
        if i % 3 == 0:
            save_checkpoint(discriminator, generator, disc_optimizer, gen_optimizer, 
                save_path=os.path.join(checkpoints_dir, f'ep_{i+1}.pth'))


def main(args):
    # Setup randomness
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    # Setup params
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    print("Train on device:", device)

    # Prepare dataloader
    dataset = Sprite16x16Dataset(args['data_root'], aug_factor=1)
    # data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_data_workers'], pin_memory=True)
    data_loader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True)

    # Instantiate and setup generator
    netG = DCGAN_Generator(hidden_size=args['hidden_size'], n_feature_maps=256, output_shape=IMAGE_SHAPE)
    netG.apply(weights_init_dcgan)

    # Instantiate and setup discriminator
    netD = DCGAN_Discriminator(input_shape=IMAGE_SHAPE, n_feature_maps=256)
    netD.apply(weights_init_dcgan)

    # Loss
    loss = nn.BCELoss().to(device)

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
    args = json.load(open(os.path.join(PROJECT_DIR, 'params/param_set1.json'), 'rb'))
    main(args)