import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from .age_networks import *
from .loss_functions import *
from .tools import *

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_epoch = 1
    rec_lambda = 1000

    age_E = Age_Net(age_enc())
    age_G = Age_Net(age_gen())

    KL_min = KL_age_loss(minimize=True)
    KL_max = KL_age_loss(minimize=False)
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()

    age_optim_E = optim.Adam(age_E.parameters(), lr=0.0002, betas=(0.5, 0.999))
    age_optim_G = optim.Adam(age_G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in num_epoch:
        for data in trainloader:
            x, label = data
            # Update encoder
            loss_E = []
            age_optim_E.zero_grad()

            z = age_E(x)
            loss_E.append(KL_min(z))

            z_sample = #TODO sample_z(sphere=True)
            x_fake = age_G(z_sample).detach()
            z_fake = age_E(x_fake)
            loss_E.append(KL_max(z_fake))

            sum(loss_E).backward()
            age_optim_E.step()

            # Update generator
            for i in range(2):
                loss_G = []
                age_optim_G.zero_grad()

                z_sample =  # TODO sample_z(sphere=True)
                x_fake = age_G(z_sample)
                z_fake = age_E(x_fake)
                loss_G.append(KL_min(z_fake))

                loss_G.append(loss_l2(z_fake, z_sample))

                sum(loss_G).backward()
                age_optim_G.step()

    with torch.no_grad():
        images = #
        for i in range(4):
            z_sample = # TODO sample_z(sphere=True)
            x_fake = age_G(z_sample)
            images = # TODO add x_fake

        im_show(torchvision.utils.make_grid(images))


