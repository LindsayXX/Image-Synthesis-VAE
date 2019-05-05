import torch
import torch.nn as nn
import torch.nn.functional as F

class Intro_Net(nn.Module):
    def __init__(self, net):
        super(Intro_Net, self).__init__()
        self.net = net

    def forward(self, input):
        output = self.net(input)
        return output

# TODO generator


# TODO encoder