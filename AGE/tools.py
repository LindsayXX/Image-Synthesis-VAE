import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

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

def age_im_gen(generator, batch_size, z_dim, device, path, data = 'celebA'):
    generator.eval()
    z_s = sampling(batch_size, z_dim).to(device)
    x_fake = generator(z_s).cpu()
    if data == 'cifar':
        vutils.save_image(x_fake / 2 + 0.5, path)
    else:
        mean1 = torch.FloatTensor([0.485, 0.456, 0.406])
        std1 = torch.FloatTensor([0.229, 0.224, 0.225])
        for i in range(list(mean1.size())[0]):
            x_fake[:, i, :, :] = x_fake[:, i, :, :].mul(std1[i]).add(mean1[i])
        vutils.save_image(x_fake.cpu(), path)
    generator.train()

def age_im_rec(encoder, generator, input, device, z_dim, path, data = 'celebA'):
    encoder.eval()
    generator.eval()
    z = encoder(input)
    x_rec = generator(z)
    t = torch.FloatTensor(input.size(0) * 2, input.size(1), input.size(2), input.size(3))
    t[0::2] = input.data.cpu()[:]
    t[1::2] = x_rec.data.cpu()[:]

    if data == 'cifar':
        vutils.save_image(t / 2 + 0.5, path)
    else:
        mean1 = torch.FloatTensor([0.485, 0.456, 0.406])
        std1 = torch.FloatTensor([0.229, 0.224, 0.225])
        for i in range(list(mean1.size())[0]):
            t[:, i, :, :] = t[:, i, :, :].mul(std1[i]).add(mean1[i])
        vutils.save_image(t.cpu(), path)
    encoder.train()
    generator.train()

def vae_im_gen(generator, batch_size, z_dim, device, path, data = 'celebA'):
    generator.eval()
    z_s = sampling(batch_size, z_dim, sphere=False, intro=True).to(device)
    x_fake = generator(z_s).cpu()
    if data == 'cifar':
        vutils.save_image(x_fake / 2 + 0.5, path)
    else:
        mean1 = torch.FloatTensor([0.485, 0.456, 0.406])
        std1 = torch.FloatTensor([0.229, 0.224, 0.225])
        for i in range(list(mean1.size())[0]):
            x_fake[:, i, :, :] = x_fake[:, i, :, :].mul(std1[i]).add(mean1[i])
        vutils.save_image(x_fake.cpu(), path)
    generator.train()

def vae_im_rec(encoder, generator, input, path, ngpu, Z_DIM, data = 'celebA'):
    encoder.eval()
    generator.eval()
    mean, logvar = encoder(input)
    z = reparameterization(mean, logvar, ngpu, opt_batch = input.size(0), opt_z_dim = Z_DIM)
    x_rec = generator(z)

    t = torch.FloatTensor(input.size(0) * 2, input.size(1), input.size(2), input.size(3))
    t[0::2] = input.data.cpu()[:]
    t[1::2] = x_rec.data.cpu()[:]

    if data == 'cifar':
        vutils.save_image(t / 2 + 0.5, path)
    else:
        mean1 = torch.FloatTensor([0.485, 0.456, 0.406])
        std1 = torch.FloatTensor([0.229, 0.224, 0.225])
        for i in range(list(mean1.size())[0]):
            t[:, i, :, :] = t[:, i, :, :].mul(std1[i]).add(mean1[i])
        vutils.save_image(t.cpu(), path)
    encoder.train()
    generator.train()

def im_show(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

