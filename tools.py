import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.utils as vutils
from torch.utils import data
from torchvision import transforms, datasets

from IntroVAE.introvae_networks import Intro_enc, Intro_gen


def load_model(IMG_DIM, Z_DIM, ngpu, model_dir, device):
    """
    Load a model
    :param IMG_DIM: Dimensions of the image
    :param Z_DIM: Latent dimensions
    :param ngpu: Number of gpus
    :param model_dir: Directory of the model
    """
    intro_E = Intro_enc(img_dim=IMG_DIM, z_dim=Z_DIM, ngpu=ngpu).to(device)
    intro_G = Intro_gen(z_dim=Z_DIM, ngpu=ngpu).to(device)
    checkpoint_E = torch.load(model_dir + 'encoder_')
    checkpoint_G = torch.load(model_dir + 'generator_')
    intro_E.load_state_dict(checkpoint_E['state_dict'])
    intro_G.load_state_dict(checkpoint_G['state_dict'])

    return intro_E, intro_G


def reparameterization(mean, logvar, ngpu, Z_DIM, device, batch_size):
    # z = mu + sigma.mul(eps)
    std = logvar.mul(0.5).exp_()
    eps = torch.FloatTensor(batch_size, Z_DIM).normal_().to(device)
    z = eps.mul(std).add_(mean)

    return z  # .unsqueeze_(-1).unsqueeze_(-1)


def load_data(dataset='celebA', root='.\data', batch_size=16, num_worker=0, imgsz=128):
    """
    :param dataset: The dataset name/type
    :param root: The root directory
    :param batch_size: The batch size of the training
    :param num_worker: Number of workers for parallel computing of the DataLoader
    :param imgsz: Image size of each sample
    :return: A trainloader object of the dataset, the model directory, the plot directory
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

    root_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    if not os.path.exists(f'age_model_{dataset}'):
        os.mkdir(f'age_model_{dataset}')
    if not os.path.exists(f'age_plot_{dataset}'):
        os.mkdir(f'age_plot_{dataset}')
    model_dir = os.path.join(root_dir, f'age_model_{dataset}')
    plot_dir = os.path.join(root_dir, f'age_plot_{dataset}')

    if dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                      pin_memory=True,
                                      drop_last=True)

    if dataset == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.SVHN(root='.\data', transform=transform, download=True)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                      pin_memory=True,
                                      drop_last=True)

    if dataset == 'celebA':
        transform = transforms.Compose([
            transforms.Resize([imgsz, imgsz]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        db = datasets.ImageFolder(root, transform=transform)
        trainloader = data.DataLoader(db, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker,
                                      drop_last=True)

    if dataset == 'celebAtest':
        transform = transforms.Compose([
            transforms.Resize([imgsz, imgsz]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        db = datasets.ImageFolder(root, transform=transform)
        indice = list(range(0, 200))
        try_sampler = data.SubsetRandomSampler(indice)
        trainloader = data.DataLoader(db, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_worker,
                                      drop_last=True, sampler=try_sampler)

    print(f'finish loading {dataset} data!')

    return trainloader, model_dir, plot_dir


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
    """
    :param z: Data in the latent dimension
    :return: Normalize M-dim z to a sphere, dim 1 is the batch size

    """

    z.div(z.norm(2, dim=1, keepdim=True).expand_as(z))

    return z


def sampling(batch_size, z_dim, sphere=True, intro=False):
    """
    Sampling function from a sphere
    :param batch_size: The batch size used
    :param z_dim: Size of the latent dimension
    :param sphere: Mapping of a normalized gaussian sphere
    :return: the samples from the batch size
    """
    if sphere:
        samples = torch.randn(batch_size, z_dim, 1, 1)
        samples = normalize(samples)
    if intro:
        samples = torch.randn(batch_size, z_dim)
        # samples = torch.FloatTensor(batch_size, z_dim)

    return samples


def age_im_gen(generator, batch_size, z_dim, device, path, data='celebA'):
    """
    Save the generated images
    :param generator: The generator model
    :param batch_size: The batch size
    :param z_dim: The latent dimension
    :param device: The device which the parameters are running (cpu or gpu)
    :param path: path to which we save the generated image
    :param data: Data type
    """

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


def age_im_rec(encoder, generator, input, path, data='celebA'):
    """
    The Age reconstruction model
    :param encoder: The encoder model
    :param generator: The generator model
    :param input:
    :param path: path to which we save the generated image
    :param data: Data type
    """
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


def vae_im_gen(generator, batch_size, z_dim, device, path, data='celebA'):
    """

    :param generator: the generator model
    :param batch_size: number of batches
    :param z_dim: latent dimension space
    :param device: what kind of device we are using (cuda or gpu)
    """
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


def vae_im_rec(encoder, generator, input, path, ngpu, Z_DIM, data='celebA'):
    """

    :param encoder: the encoder model
    :param generator: the generator model
    :param input: the actual input
    :param path: the path of the saved modefl
    :param ngpu: number of gpu
    :param Z_DIM: latent dimensional space
    """
    encoder.eval()
    generator.eval()
    mean, logvar = encoder(input)
    z = reparameterization(mean, logvar, ngpu, opt_batch=input.size(0), opt_z_dim=Z_DIM)
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
    """
    Plot of a single image.
    :param img: the image
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
