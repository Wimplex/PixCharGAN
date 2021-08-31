import torch.nn as nn


def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN_Generator(nn.Module):
    def __init__(self, hidden_size=128, output_shape=(3, 16, 16), n_feature_maps=64):
        super(DCGAN_Generator, self).__init__()
        self.ct1 = nn.ConvTranspose2d(hidden_size, n_feature_maps * 8, 4, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_feature_maps * 8)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.ct2 = nn.ConvTranspose2d(n_feature_maps * 8, n_feature_maps * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_feature_maps * 4)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.ct3 = nn.ConvTranspose2d(n_feature_maps * 4, n_feature_maps * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(n_feature_maps * 2)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.ct4 = nn.ConvTranspose2d(n_feature_maps * 2, n_feature_maps, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(n_feature_maps)
        self.relu4 = nn.LeakyReLU(inplace=True)

        self.ct5 = nn.ConvTranspose2d(n_feature_maps, output_shape[0], 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.ct1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.ct2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.ct3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.ct4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.ct5(x)
        out = self.tanh(x)
        return out


class DCGAN_Discriminator(nn.Module):
    def __init__(self, input_shape=[16, 16, 3], n_feature_maps=64):
        super(DCGAN_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], n_feature_maps, 4, 2, 1, bias=False)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(n_feature_maps, n_feature_maps * 2, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_feature_maps * 2)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(n_feature_maps * 2, n_feature_maps * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_feature_maps * 4)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(n_feature_maps * 4, n_feature_maps * 8, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(n_feature_maps * 8)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(n_feature_maps * 8, 1, 2, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.lrelu2(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.lrelu3(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.lrelu4(x)

        x = self.conv5(x)
        out = self.sigmoid(x)
        return out

