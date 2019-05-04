import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from .age_networks import *
from .loss_functions import *
from .tools import *

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indice = list(range(0, 1000))
    trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, sampler=data.SubsetRandomSampler(indice))
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #testset_sub = torch.utils.data.SubsetRandomSampler(indice)
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    NUM_EPOCH = 1
    REC_LAMBDA = 1000
    Z_DIM = 128

    age_E = Age_Net(age_enc())
    age_G = Age_Net(age_gen())

    KL_min = KL_Loss_AGE(minimize=True)
    KL_max = KL_Loss_AGE(minimize=False)
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()

    age_optim_E = optim.Adam(age_E.parameters(), lr=0.0002, betas=(0.5, 0.999))
    age_optim_G = optim.Adam(age_G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in NUM_EPOCH:
        for i, data in enumerate(trainloader, 0):
            x, label = data
            batch_size = list(x.size())[0]
            # Update encoder
            loss_E = []
            age_optim_E.zero_grad()

            z = age_E(x)
            loss_E.append(KL_min(z))

            z_sample = sampling(batch_size, Z_DIM)
            x_fake = age_G(z_sample).detach()
            z_fake = age_E(x_fake)
            loss_E.append(KL_max(z_fake))

            sum(loss_E).backward()
            age_optim_E.step()

            # Update generator
            for g_i in range(2):
                loss_G = []
                age_optim_G.zero_grad()

                z_sample = sampling(batch_size, Z_DIM)
                x_fake = age_G(z_sample)
                z_fake = age_E(x_fake)
                loss_G.append(KL_min(z_fake))

                loss_G.append(REC_LAMBDA * loss_l2(z_fake, z_sample))

                sum(loss_G).backward()
                age_optim_G.step()

    with torch.no_grad():
        for i in range(4):
            z_sample = sampling(batch_size, Z_DIM)
            x_fake = age_G(z_sample)
            im_show(torchvision.utils.make_grid(x_fake))


