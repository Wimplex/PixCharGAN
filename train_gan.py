import warnings
warnings.filterwarnings('ignore')

import os
import random
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

from dataloader import Sprite16x16Dataset
from networks.dcgan import DCGAN_Generator, DCGAN_Discriminator, weights_init_dcgan
from utils import save_checkpoint, draw_img
from config import Config


# Instantiate globals
train_writer = SummaryWriter()
config = Config()


def train_one_epoch(generator: torch.nn.Module, discriminator: torch.nn.Module, \
    train_loader: DataLoader, gen_optimizer: optim.Optimizer, disc_optimizer: optim.Optimizer, \
    criterions: dict, device: str, hidden_size: int, current_epoch: int):
    
    for i, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        x_batch, direction_batch, _ = data
        real_batch = x_batch.to(device)
        direction_batch = direction_batch.to(device)
        batch_size = real_batch.shape[0]

        # Discriminator training. The data entirely is real (accumulate gradients)
        discriminator.zero_grad()
        label = torch.full([batch_size,], config.REAL_LABEL, dtype=torch.float, device=device)
        output = discriminator(real_batch).view(-1)
        loss_D_real = criterions['bce'](output, label)
        loss_D_real.backward()

        # Generate fake data
        noise = torch.randn(batch_size, hidden_size, device=device)
        fake_batch = generator(noise, direction_batch)

        # Discriminator training. The data entirely is fake (accumulate gradients).
        label.fill_(config.FAKE_LABEL)
        output = discriminator(fake_batch.detach()).view(-1)
        loss_D_fake = criterions['bce'](output, label)
        loss_D_fake.backward()
        disc_optimizer.step()

        # Generator training. Accumulate gradients for generator
        generator.zero_grad()
        label.fill_(config.REAL_LABEL)
        output = discriminator(fake_batch).view(-1)
        loss_G = criterions['bce'](output, label)
        loss_G.backward()
        gen_optimizer.step()

        # Log losses
        curr_iter = current_epoch * len(train_loader) + i
        train_writer.add_scalars('pix_gan_losses', {
            'gen_loss': loss_G.detach().item(),
            'disc_loss': loss_D_real.detach().item() + loss_D_fake.detach().item()
        }, curr_iter)


def train_one_epoch_l2_objective(
    generator: torch.nn.Module, discriminator: torch.nn.Module,
    train_loader: DataLoader, 
    gen_optimizer: optim.Optimizer, disc_optimizer: optim.Optimizer,
    criterions: dict,
    device: str, hidden_size: int, current_epoch: int
    ):

    for i, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        tensor_batch, direction_batch, _ = data
        real_batch = tensor_batch.to(device)
        direction_batch = direction_batch.to(device)

        # Discriminator training. The data entirely is real (accumulate gradients)
        discriminator.zero_grad()
        batch_size = real_batch.size(0)
        label = torch.full([batch_size,], config.REAL_LABEL, dtype=torch.float, device=device, requires_grad=True)
        output, emb_real = discriminator(real_batch.detach())
        loss_D_real = criterions['bce'](output.view(-1), label.detach())
        loss_D_real.backward()

        # Batch of fake images generation
        noise = torch.randn(batch_size, hidden_size, device=device)
        fake_batch = generator(noise, direction_batch)

        # Discriminator training. The data entirely is fake (accumulate gradients)
        label = torch.full([batch_size,], config.FAKE_LABEL, dtype=torch.float, device=device, requires_grad=True)
        output, _ = discriminator(fake_batch.detach())
        loss_D_fake = criterions['bce'](output.view(-1), label.detach())
        loss_D_fake.backward()

        # Discriminator update
        disc_optimizer.step()

        # Generator training. The data trying to pretend to be real
        generator.zero_grad()
        label = torch.full([batch_size,], config.REAL_LABEL, dtype=torch.float, device=device, requires_grad=True)
        output, emb_fake = discriminator(fake_batch.detach())
        # loss_G_penalize = torch.mm(F.normalize(emb_fake), F.normalize(emb_real).T.detach()).diag().mean()
        loss_G_penalize = criterions['cosine'](emb_fake, emb_real.detach()).mean()
        loss_G = criterions['bce'](output.view(-1), label.detach()) + 0.1 * loss_G_penalize
        loss_G.backward()

        # Generator update
        gen_optimizer.step()

        # Losses logging
        curr_iter = current_epoch * len(train_loader) + i
        train_writer.add_scalars('pix_gan_losses', {
            'gen_loss': loss_G.detach().item(),
            'disc_loss': loss_D_real.detach().item() + loss_D_fake.detach().item(),
        }, curr_iter)


def train(generator: nn.Module, discriminator: nn.Module, train_loader: DataLoader, \
    num_epochs: int, gen_optimizer: optim.Optimizer, disc_optimizer: optim.Optimizer, \
    criterions: dict, device: str, hidden_size: int):

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
    checkpoints_dir = os.path.join(config.PROJECT_DIR, 'checkpoints', curr_time)
    os.makedirs(checkpoints_dir, exist_ok=True)

    for i in range(num_epochs):
        print(f"Epoch {i + 1} started.")
        train_one_epoch(
        # train_one_epoch_l2_objective(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            criterions=criterions,
            device=device,
            hidden_size=hidden_size,
            current_epoch=i
        )
        
        # Plot and save visualization
        with torch.no_grad():
            noise = fixed_data['noise']
            dimensions = fixed_data['dimensions']
            fake = generator(noise, dimensions).detach().cpu()
            img_grid = vutils.make_grid(fake, padding=2, normalize=True).detach().cpu().numpy()
            draw_img(img_grid, os.path.join(config.OUTPUT_PLOTS_DIR, f'{i+1}.png'))

        # Save checkpoint
        if config.SAVE_CHECKPOINT_EVERY is not None and i % config.SAVE_CHECKPOINT_EVERY == 0:
            save_checkpoint(discriminator, generator, disc_optimizer, gen_optimizer, 
                save_path=os.path.join(checkpoints_dir, f'ep_{i+1}.pth'))
            config.save_config(os.path.join(checkpoints_dir, 'params.json'))


def main(new_config=None):

    # Replace config
    if new_config is not None:
        global config
        config = new_config

    # Setup randomness
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # Setup params
    torch.backends.cudnn.benchmark = True
    print(f"Train on {config.DEVICE}")

    # Prepare dataloader
    dataset = Sprite16x16Dataset(config.DATA_ROOT, aug_factor=1, max_pad_size=config.IMAGE_SHAPE[0])
    # -- shuffle=True/num_workers>1/pin_memory=True - casuses iterative RAM overconsumption
    # data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=args['num_data_workers'], pin_memory=True)
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Create and setup generator
    netG = DCGAN_Generator(hidden_size=config.HIDDEN_SIZE, n_feature_maps=128, output_shape=config.IMAGE_SHAPE)
    netG.apply(weights_init_dcgan)

    # Create and setup discriminator
    netD = DCGAN_Discriminator(input_shape=config.IMAGE_SHAPE, n_feature_maps=128)
    netD.apply(weights_init_dcgan)

    # Loss
    bce_loss = nn.BCELoss().to(config.DEVICE)
    cosine_loss = nn.CosineSimilarity(dim=0).to(config.DEVICE)
    criterions = {'bce': bce_loss, 'cosine': cosine_loss}

    # Optimizers
    optG = optim.Adam(netG.parameters(), lr=config.GENERATOR_LR, betas=[config.BETA1, 0.999])
    optD = optim.Adam(netD.parameters(), lr=config.DISCRIMINATOR_LR, betas=[config.BETA1, 0.999])

    # Start training process
    train(
        generator=netG,
        discriminator=netD,
        num_epochs=config.NUM_EPOCHS,
        gen_optimizer=optG,
        disc_optimizer=optD,
        train_loader=data_loader,
        criterions=criterions,
        device=config.DEVICE,
        hidden_size=config.HIDDEN_SIZE
    )


if __name__ == '__main__':
    main()