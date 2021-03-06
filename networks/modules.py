import torch
import torch.nn as nn


activations = nn.ModuleDict([
    ['relu', nn.ReLU(True)],
    ['lrelu', nn.LeakyReLU(0.2, True)],
    ['tanh', nn.Tanh()],
    ['selu', nn.SELU(True)],
    ['sigmoid', nn.Sigmoid()]
])


class PrintOutputShape(nn.Module):
    def __init__(self, prefix=''):
        super(PrintOutputShape, self).__init__()
        self.prefix = prefix

    def forward(self, x):
        print(self.prefix, x.shape)
        return x


class Reshape(nn.Module):
    def __init__(self, target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return torch.reshape(x, [x.shape[0]] + list(self.target_shape))


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=1, bias=False, \
        activation='lrelu', bn=True, dropout=None):
        super(Conv2dBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.Dropout2d(dropout) if dropout is not None else nn.Identity(),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            activations[activation]
        )

    def forward(self, x):
        return self.blocks(x)


class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=1, bias=False, \
        activation='lrelu', bn=True, dropout=None):
        super(ConvTranspose2dBlock, self).__init__()
        self.blocks = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=bias),
            nn.Dropout2d(dropout) if dropout is not None else nn.Identity(),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            activations[activation]
        )

    def forward(self, x):
        return self.blocks(x)
