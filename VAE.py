from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from AGE.loss_functions import *
from AGE.tools import *
import os
import sys

class VAE(nn.module):
    def __init__(self):
        super(VAE, self).__init__()

    def forward(self, input):

        return output