import torch
import torch.nn as nn
from torch.utils.data.dataloader import T
import torch.nn.functional as F


def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN_Generator(nn.Module):
    def __init__(self, hidden_size=128, output_shape=[32, 32, 3], n_feature_maps=64, n_classes=4):
        super(DCGAN_Generator, self).__init__()
        self.output_shape = output_shape

        # Conditional encoding pipe
        self.cond_pipe = nn.Sequential(
            nn.Linear(n_classes, 50),
            nn.Linear(50, 30 * 2 * 2)
        )
        self.fc1 = nn.Linear(hidden_size, 30 * 2 * 2)

        # Main pipe layers
        self.main_pipe = nn.Sequential(
            nn.ConvTranspose2d(30 * 2, n_feature_maps * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(n_feature_maps * 8, n_feature_maps * 4, 4, 1, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(n_feature_maps * 4, n_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(n_feature_maps * 2, n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(n_feature_maps, output_shape[2], 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, label):
        cond_features = self.cond_pipe(label).reshape([label.shape[0], 30, 2, 2])
        x = F.leaky_relu(self.fc1(x), 0.2, True)
        x = torch.cat((x.reshape(x.shape[0], 30, 2, 2), cond_features), 1)
        out = self.main_pipe(x)
        return out


class DCGAN_Discriminator(nn.Module):
    def __init__(self, input_shape=[32, 32, 3], n_feature_maps=64, n_classes=4):
        super(DCGAN_Discriminator, self).__init__()
        self.input_shape = input_shape

        # Conditional encoding pipe
        self.cond_pipe = nn.Sequential(
            nn.Linear(n_classes, 50),
            nn.Linear(50, input_shape[0] * input_shape[1])
        )

        # Main layers
        self.main_pipe = nn.Sequential(
            nn.Conv2d(input_shape[2] + 1, n_feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feature_maps, n_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feature_maps * 2, n_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feature_maps * 4, n_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_feature_maps * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, label):
        cond_features = self.cond_pipe(label).reshape([label.shape[0], 1, self.input_shape[0], self.input_shape[1]])
        x = torch.cat((x, cond_features), 1)
        out = self.main_pipe(x)
        return out