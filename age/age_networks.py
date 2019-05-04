import torch
import torch.nn as nn
import torch.nn.functional as F

class Age_Net(nn.Module):
    def __init__(self, net, sphere):
        super(Age_Net, self).__init__()
        self.net = net
        self.sphere = sphere

    def forward(self, input):
        output = self.net(input)
        if self.sphere:
            pass # normalize
        return output

def age_enc(im_dim=32, num_col=3, z_dim=128, ndf=64, sphere=True):
    if im_dim==32:
        # Input num_col x 32 x 32
        net = nn.Sequential(
        nn.Conv2d(num_col, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 4, z_dim, 4, 2, 1, bias=True),
        nn.AvgPool2d(2))
        # Output z_dim x 1 x 1
    elif im_dim==128:
        # Input num_col x 128 x 128
        net = nn.Sequential(
        nn.Conv2d(num_col, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 16),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 16, z_dim, 4, 1, 0, bias=True))
        # Output z_dim x 1 x 1
    return Age_Net(net, sphere)

def age_gen(im_dim=32, num_col=3, z_dim=128, ngf=64):
    if im_dim==32:
        # Input z_dim
        net = nn.Sequential(
        nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf * 2, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),

        nn.Conv2d(ngf * 2, num_col, 1, bias=True),
        nn.Tanh())
        # Output num_col x 32 x 32
    elif im_dim==128:
        # Input z_dim
        net = nn.Sequential(
        nn.ConvTranspose2d(z_dim, ngf * 16, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 16),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf, num_col, 4, 2, 1, bias=False),
        nn.Tanh())
        # Output num_col x 128 x 128
    return Age_Net(net)



