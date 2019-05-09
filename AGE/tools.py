import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    '''
    Custom weights initialization called on netG and netE
    Code from AGE github
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def normalize(z):
    # normalize M-dim z to a sphere
    # dim 1 is the batch size
    z.div(z.norm(2, dim=1, keepdim=True).expand_as(z))

    return z

def sampling(batch_size, z_dim, sphere=True, intro=False):
    if sphere:
        samples = torch.randn(batch_size, z_dim, 1, 1)
        samples = normalize(samples)
    if intro:
        samples = torch.randn(batch_size, z_dim)
        #samples = torch.FloatTensor(batch_size, z_dim)

    return samples

def im_gen():
    #TODO generate images given model

    return 0

def im_show(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

