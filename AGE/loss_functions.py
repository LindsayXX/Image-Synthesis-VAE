import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import *

def cos_loss(x, y):
    x_n = normalize(x)
    y_n = normalize(y)
    return 2 - (x_n).mul(y_n).mean()

def l1_loss(x, y):
    return (x - y).abs().mean()

class KL_Loss_AGE(nn.Module):
    #age_loss
    def __init__(self, minimize):
        super(KL_Loss_AGE, self).__init__()
        self.minimize = minimize
        self.mean = 0
        self.var = 0
        self.M = 0

    def forward(self, z):
        # Input normalized z
        self.M = list(z.size())[1] # size of latent space
        self.mean = z.mean(0)
        self.var = z.var(0, unbiased=False)
        kl_loss = -self.M/2 + ((self.mean.pow(2) + self.var)/2 - self.var.sqrt().log()).sum()

        if not self.minimize:
            kl_loss *= -1

        return kl_loss

'''
# rest part of the loss function: reconstruction loss
loss_l1 = nn.L1Loss()
loss = loss_l1(recon, samples)
loss_l2 = nn.MSELoss()
loss = loss_l2(recon, samples)
'''

class KL_Loss_Intro(nn.Module):
    # IntroVAE_loss
    def __init__(self, minimize):
        super(KL_Loss_Intro, self).__init__()
        self.minimize = minimize
        self.mean = 0
        self.var = 0
        self.M = 0
        self.N = 0

    def forward(self, mean, var):
        # Input mean and variance of z
        self.M, self.N = list(mean.size())
        self.mean = mean
        self.var = var
        kl_loss = - self.N * self.M/2 + ((self.mean.pow(2) + self.var)/2 - self.var.sqrt().log()).sum()

        if not self.minimize:
            kl_loss *= -1

        return kl_loss

"""
# rest part of the loss function : loss of AutoEncoder
loss_l2 = nn.MSELoss()
loss_AE = 1/2 * loss_l2(recon, samples)
"""
