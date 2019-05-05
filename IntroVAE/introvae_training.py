from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from IntroVAE.intro_networks import *
from AGE.loss_functions import *
from AGE.tools import *

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset.data = trainset.data[np.where(np.array(trainset.targets) == 1)]  # Only cars
    # indice = list(range(0, 10000))
    # sampler=data.SubsetRandomSampler(indice)
    # trainset = torchvision.datasets.SVHN(root='.\data', transform=transform, download =True)
    trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # testset_sub = torch.utils.data.SubsetRandomSampler(indice)
    # testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    NUM_EPOCH = 10

    cudnn.benchmark = True

    # TODO: read data

    # TODO: training procedure
    for epoch in tqdm(range(NUM_EPOCH)):
        for i, data in enumerate(trainloader, 0):

    # TODO: visualize some example
