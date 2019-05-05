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

    NUM_EPOCH = 10

    # TODO: read data

    # TODO: training procedure
    #for epoch in tqdm(range(NUM_EPOCH)):
        #for i, data in enumerate(trainloader, 0):

    # TODO: visualize some example
