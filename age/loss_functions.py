import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.var = torch.var(z, False)
        kl_loss = -self.M/2 + ((self.mean.pow(2) + self.var.pow(2))/2 - (self.var.sqrt).log).sum

        if not self.minimize:
            kl_loss *= -1

        return kl_loss

'''
# rest part of the loss function: reconstruction loss
loss_l1 = nn.L1Loss()
loss = loss_l1(recon, samples)
loss_l2 = nn.MSELoss()
loss = loss_l1(recon, samples)
'''

class KL_Loss_Intro(nn.Module):

    # IntroVAE_loss
    def __init__(self, minimize):
        super(KL_Loss_Intro, self).__init__()
        self.minimize = minimize
        self.mean = 0
        self.var = 0
        self.M = 0

    def forward(self, z):
        # Input normalized z
        # TODO change this
        self.M = list(z.size())[1]  # size of latent space
        self.mean = z.mean(0)
        self.var = torch.var(z, False)
        kl_loss = -self.M / 2 + ((self.mean.pow(2) + self.var.pow(2)) / 2 - (self.var.sqrt).log).sum

        if not self.minimize:
            kl_loss *= -1

        return kl_loss


