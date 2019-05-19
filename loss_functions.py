import torch.nn as nn



def cos_loss(x, y):
    return 2 - (x).mul(y).mean()


def l1_loss(x, y):
    return (x - y).abs().mean()


def l2_loss(x, y, age=True):
    if age:  # 1/(M*N)
        loss = (x - y).pow(2)
    else:  # 1/N
        loss = (x - y).pow(2).view(list(x.size())[0], -1).sum(dim=-1)

    return loss.mean()


class KL_Loss_AGE(nn.Module):
    """
    Kullback–Leibler Loss for the AGE model
    """

    def __init__(self, minimize):
        """

        :param minimize: Boolean parameter to change between minimizing and maximizing the Loss.
        """
        super(KL_Loss_AGE, self).__init__()
        self.minimize = minimize
        self.mean = 0
        self.var = 0
        self.M = 0

    def forward(self, z):
        # Input normalized z
        """
        :param z: The input/output (depends on the occasion) in the latent dimensional space
        :return: The loss of the data
        """
        self.M = list(z.size())[1]  # size of latent space
        self.mean = z.mean(dim=0)
        self.var = z.var(dim=0, unbiased=False)
        kl_loss = -1 / 2 + ((self.mean.pow(2) + self.var) / 2 - self.var.sqrt().log()).mean()

        if not self.minimize:
            kl_loss *= -1

        return kl_loss


class KL_Loss_Intro(nn.Module):
    """
       Kullback–Leibler Loss for the IntroVAE model
    """

    def __init__(self, minimize):
        """

        :param minimize: Boolean parameter to change between minimizing and maximizing the Loss.

        """
        super(KL_Loss_Intro, self).__init__()
        self.minimize = minimize
        self.mean = 0
        self.var = 0
        self.M = 0
        self.N = 0

    def forward(self, mean, logvar):
        """

        :param mean: The mean of the input in the latent space
        :param logvar: The variance of the input in the latent space
        :return:
        """

        # Input mean and variance of z
        self.M, self.N = list(mean.size())
        self.mean = mean
        self.logvar = logvar
        var = logvar.exp()
        kl_loss = (-1 - self.logvar + self.mean.pow(2) + var).mul_(0.5).sum(dim=-1)
        kl_loss = kl_loss.mean()

        if not self.minimize:
            kl_loss *= -1

        return kl_loss
