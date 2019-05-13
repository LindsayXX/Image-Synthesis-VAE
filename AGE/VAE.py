from __future__ import print_function
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms, datasets
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os
import sys
from torchvision.utils import save_image
import numpy as np
import random
from age_training import *

#from dlutils.pytorch.cuda_helper import *

"""
Network
"""
class VAE(nn.Module):
    def __init__(self, zsize, layer_count=3, channels=3):
        super(VAE, self).__init__()

        d = 128
        self.d = d
        self.zsize = zsize
        self.layer_count = layer_count
        mul = 1
        inputs = channels
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul *= 2

        self.d_max = inputs

        self.fc1 = nn.Linear(inputs * 4 * 4, zsize)
        self.fc2 = nn.Linear(inputs * 4 * 4, zsize)

        self.d1 = nn.Linear(zsize, inputs * 4 * 4)

        mul = inputs // d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))

    def encode(self, x):
        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))

        x = x.view(x.shape[0], self.d_max * 4 * 4)
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = x.view(x.shape[0], self.zsize)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, 4, 4)
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)

        x = torch.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


"""
training 
"""
im_size = 128
cudnn.benchmark = True
random.seed(123)

def loss_function(recon_x, x, mu, logvar):
    BCE = torch.mean((recon_x - x) ** 2)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1


def main(train_epoch=40, lr=0.0005, batch_size=2, z_size=512, im_size=128, m=60):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f'Cuda available: {torch.cuda.is_available()}')
        print(f'Current device number: {torch.cuda.current_device()}')
        print(f'Current device: {torch.cuda.device(torch.cuda.current_device())}')
        print(f'Number of GPUs: {torch.cuda.device_count()}')
        print(f'Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        print(f'Used device: {device}')
        ngpu = torch.cuda.device_count()
        root = os.path.abspath(os.path.dirname(sys.argv[0])) + '/../data/'
        trainloader, model_dir, plot_dir = load_data('celebA', root=root, batch_size=batch_size, num_worker=0,
                                                     imgsz=im_size)
    else:
        device = torch.device('cpu')
        print(f'Used device: {device}')
        ngpu = 0
        root = 'D:\MY1\DPDS\project\DD2424-Projekt\data'
        trainloader, model_dir, plot_dir = load_data('celebAtest', root=root, batch_size=batch_size, num_worker=0,
                                                     imgsz=im_size)

    x = torch.FloatTensor(batch_size, 3, im_size, im_size).to(device)
    x = Variable(x)

    vae = VAE(zsize=z_size, layer_count=5).to(device)
    #vae.cuda()
    vae.train()
    vae.weight_init(mean=0, std=0.02)

    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

    sample1 = torch.randn(128, z_size).view(-1, z_size, 1, 1)

    os.makedirs('results_rec', exist_ok=True)
    os.makedirs('results_gen', exist_ok=True)

    for epoch in tqdm(range(train_epoch)):
        vae.train()
        rec_loss = 0
        kl_loss = 0

        if (epoch + 1) % 8 == 0:
            vae_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        #i = 0
        for i, data in enumerate(trainloader, 0):
            input, label = data
            x.data.copy_(input)
            vae.train()
            vae.zero_grad()
            rec, mu, logvar = vae(x)

            loss_re, loss_kl = loss_function(rec, x, mu, logvar)
            (loss_re + loss_kl).backward()
            vae_optimizer.step()
            rec_loss += loss_re.item()
            kl_loss += loss_kl.item()

            #############################################
            #os.makedirs('results_rec', exist_ok=True)
            #os.makedirs('results_gen', exist_ok=True)

            # report losses and save samples each m(60) iterations
            #i += 1
            if i % m == 0:
                rec_loss /= m
                kl_loss /= m
               # print('\n[%d/%d] - ptime: %.2f, rec loss: %.9f, KL loss: %.9f' % (
                    #(epoch + 1), train_epoch, per_epoch_ptime, rec_loss, kl_loss))
                print('\n[%d/%d]rec loss: %.7f, KL loss: %.7f' %((epoch + 1),train_epoch,rec_loss, kl_loss))
                rec_loss = 0
                kl_loss = 0
                with torch.no_grad():
                    vae.eval()
                    x_rec, _, _ = vae(x)
                    resultsample = torch.cat([x, x_rec]) * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_rec/sample_' + str(epoch) + "_" + str(i) + '.png')
                    x_rec = vae.decode(sample1)
                    resultsample = x_rec * 0.5 + 0.5
                    resultsample = resultsample.cpu()
                    save_image(resultsample.view(-1, 3, im_size, im_size),
                               'results_gen/sample_' + str(epoch) + "_" + str(i) + '.png')

    print("Training finish!... save training results")
    torch.save(vae.state_dict(), "VAEmodel.pkl")

if __name__ == '__main__':
    main()