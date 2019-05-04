import torch
import torch.nn as nn
import torch.nn.functional as F

class KL_age_loss(nn.Module):

    def __init__(self, minimize):
        super(Net, self).__init__()
        self.minimize = minimize

    def forward(self, z):
        # Input normalized z

        return loss