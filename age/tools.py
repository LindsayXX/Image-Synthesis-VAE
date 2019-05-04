import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(z):
    # normalize M-dim z to a sphere
    # dim 1 is the batch size
    z.div(z.norm(2, dim=1, keepdim=True).expand_as(z))

    return z

def sampling(batch_size, z_dim, sphere=True):
    samples = torch.randn(batch_size, z_dim, 1, 1)
    if sphere:
        samples = normalize(samples)

    return samples

def im_gen():
    #TODO generate images given model

    return 0

def im_show(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
