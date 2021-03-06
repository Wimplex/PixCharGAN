import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.modules import *


def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname == 'Conv2d' or classname == 'ConvTranspose2d':
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN_Generator(nn.Module):
    def __init__(self, hidden_size=128, output_shape=[32, 32, 4], n_feature_maps=64, num_classes=4):
        super(DCGAN_Generator, self).__init__()

        # Shape of output image
        self.output_shape = output_shape

        # Size of main embedding Z
        self.hidden_size = hidden_size

        # Shape of feature maps, reshaped from embedding Z
        self.hidden_shape = 2

        # Conditional encoding pipe
        self.cond_pipe = nn.Sequential(
            nn.Embedding(num_classes, 50),
            nn.Linear(50, self.hidden_shape ** 2)
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size * self.hidden_shape ** 2)
        self.lrelu = nn.LeakyReLU(0.2, True)

        # Main pipe layers
        self.main_pipe = nn.Sequential(
            ConvTranspose2dBlock(hidden_size + num_classes, n_feature_maps * 8, 4, 2),
            ConvTranspose2dBlock(n_feature_maps * 8, n_feature_maps * 4, 4, 2),
            ConvTranspose2dBlock(n_feature_maps * 4, n_feature_maps * 2, 4, 2),
            ConvTranspose2dBlock(n_feature_maps * 2, n_feature_maps, 4, 2),
            ConvTranspose2dBlock(n_feature_maps, output_shape[2], 4, 1, activation='tanh', bn=False)
        )

    def forward(self, x, condition):

        # Extract conditional embedding
        cond_features = self.cond_pipe(condition)
        cond_features = torch.reshape(
            cond_features, 
            [x.shape[0], cond_features.shape[1], self.hidden_shape, self.hidden_shape]
        )

        # Apply to main embedding Z
        x = self.fc1(x)
        x = torch.reshape(x, [x.shape[0], self.hidden_size, self.hidden_shape, self.hidden_shape])
        x = F.leaky_relu(torch.cat((x, cond_features), 1), 0.2, True)
        out = self.main_pipe(x)
        return out


class DCGAN_Discriminator(nn.Module):
    def __init__(self, input_shape=[32, 32, 4], n_feature_maps=64, emb_size=96):
        super(DCGAN_Discriminator, self).__init__()
        self.input_shape = input_shape

        # Main layers
        self.main_pipe = nn.Sequential(
           Conv2dBlock(input_shape[2], n_feature_maps, 4, 2, bn=False),
           Conv2dBlock(n_feature_maps, n_feature_maps * 2, 4, 2),
           Conv2dBlock(n_feature_maps * 2, n_feature_maps * 4, 4, 2),
           Conv2dBlock(n_feature_maps * 4, n_feature_maps * 8, 4, 2),
           Conv2dBlock(n_feature_maps * 8, 1, 2, 1, padding=0, activation='sigmoid', bn=False),
        )

        #self.main_pipe_with_intermediate_feature_layer = nn.Sequential(
        #    Conv2dBlock(input_shape[2], n_feature_maps, 4, 2, bn=False),
        #    Conv2dBlock(n_feature_maps, n_feature_maps * 2, 4, 2),
        #    Conv2dBlock(n_feature_maps * 2, n_feature_maps * 4, 4, 2),
        #    Conv2dBlock(n_feature_maps * 4, n_feature_maps * 8, 4, 2),
        #    Reshape([2 * 2 * n_feature_maps * 8]),
        #    nn.Linear(2 * 2 * n_feature_maps * 8, emb_size)
        #)
        #self.end_of_pipe = nn.Sequential(
        #    nn.Linear(emb_size, 1),
        #    nn.Sigmoid()
        #)

    def forward(self, x):
        out = self.main_pipe(x)
        return out

        # feats = self.main_pipe_with_intermediate_feature_layer(x)
        # out = self.end_of_pipe(feats)
        # return out