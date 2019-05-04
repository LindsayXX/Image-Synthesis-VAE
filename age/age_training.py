import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from .age_networks import *
from .loss_functions import *

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    KL_min = KL_age_loss(minimize=True)
    KL_max = KL_age_loss(minimize=False)
    age_E = Age_Net(age_enc())
    age_G = Age_Net(age_gen())
    age_optim_E = optim.Adam(age_E.parameters(), lr=0.0002, betas=(0.5, 0.999))
    age_optim_G = optim.Adam(age_G.parameters(), lr=0.0002, betas=(0.5, 0.999))