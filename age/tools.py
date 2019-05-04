import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize():
    #TODO normalize M-dim z to sphere
    return 0

def z_sample():
    # TODO sample z

def im_gen():
    #TODO generate images given model
    return 0

def im_show(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
