from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
from age_networks import *
from loss_functions import *
from tools import *
import sys
import os
import matplotlib.pyplot as plt
import random
import numpy as np

def load_data(dataset='celebA', root='.\data', batch_size=16, num_worker=0, imgsz=128):
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
        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # trainset.data = trainset.data[np.where(np.array(trainset.targets)==1)] # Only cars
        # indice = list(range(0, 10000))
        # sampler=data.SubsetRandomSampler(indice)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True,
                                      drop_last=True)
        # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        # testset_sub = torch.utils.data.SubsetRandomSampler(indice)
        # testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    if dataset == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.SVHN(root='.\data', transform=transform, download=True)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True,
                                      drop_last=True)

    if dataset == 'celebA':
        transform = transforms.Compose([
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize([imgsz, imgsz]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        db = datasets.ImageFolder(root, transform=transform)
        #train_sampler = data.SubsetRandomSampler(list(range(0, 29000)))
        trainloader = data.DataLoader(db, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_worker, drop_last=True)

    if dataset == 'celebAtest':
        transform = transforms.Compose([
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize([imgsz, imgsz]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        db = datasets.ImageFolder(root, transform=transform)
        indice = list(range(0, 200))
        try_sampler = data.SubsetRandomSampler(indice)
        # train_sampler = data.SubsetRandomSampler(list(range(0, 29000)))
        trainloader = data.DataLoader(db, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_worker,
                                      drop_last=True, sampler=try_sampler)

    print(f'finish loading {dataset} data!')

    return trainloader, model_dir, plot_dir

if __name__ == '__main__':
    cudnn.benchmark = True
    random.seed(123)
    #REC_LAMBDA = 1000
    #REC_MU = 10
    LR = 0.0002
    # settings for cifar
    '''
    NUM_EPOCH = 150
    Z_DIM = 128
    DROP_LR = 40
    batch_size = 64
    IM_DIM = 32
    G_UPDATES = 2
    save_model = 15
    SAMPLE_BATCH = 64
    '''

    # setting for celebA
    NUM_EPOCH = 100#
    REC_LAMBDA = 1000
    REC_MU = 10
    Z_DIM = 64
    DROP_LR = 50#50
    batch_size = 8
    IM_DIM = 128
    G_UPDATES = 3
    save_model = 10
    SAMPLE_BATCH = 16

    START_EPOCH = 2
    LOAD_MODEL = True

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
        # trainloader, model_dir, plot_dir = load_data('celebA', root=root, batch_size=batch_size, num_worker=0, imgsz=IM_DIM)
        trainloader, model_dir, plot_dir = load_data('celebA', root=root, batch_size=batch_size, num_worker=0,
                                                     imgsz=IM_DIM)

    else:
        device = torch.device('cpu')
        print(f'Used device: {device}')
        ngpu = 0
        root = 'D:\MY1\DPDS\project\DD2424-Projekt\data'
        # root = 'C:/Users/Alexander/Desktop/Skolgrejs/deep/project/DD2424-Projekt/data/'
        trainloader, model_dir, plot_dir = load_data('celebAtest', root=root, batch_size=batch_size, num_worker=0,
                                                     imgsz=IM_DIM)


    age_E = age_enc(z_dim=Z_DIM, ngpu=ngpu, im_dim=IM_DIM).to(device)
    age_G = age_gen(z_dim=Z_DIM, ngpu=ngpu, im_dim=IM_DIM).to(device)

    x = torch.FloatTensor(batch_size, 3, IM_DIM, IM_DIM).to(device)
    z_sample = torch.FloatTensor(batch_size, Z_DIM, 1, 1).to(device)
    x = Variable(x)
    z_sample = Variable(z_sample)

    KL_min = KL_Loss_AGE(minimize=True)
    KL_max = KL_Loss_AGE(minimize=False)
    #loss_l1 = nn.L1Loss()
    #loss_l2 = nn.MSELoss()

    age_optim_E = optim.Adam(age_E.parameters(), lr=LR, betas=(0.5, 0.999))
    age_optim_G = optim.Adam(age_G.parameters(), lr=LR, betas=(0.5, 0.999))

    if not LOAD_MODEL:
        age_E.apply(weights_init)
        age_G.apply(weights_init)
    else:
        checkpoint_E = torch.load(f"{model_dir}/encoder_{START_EPOCH - 1}")
        checkpoint_G = torch.load(f"{model_dir}/generator_{START_EPOCH - 1}")
        age_E.load_state_dict(checkpoint_E['state_dict'])
        age_G.load_state_dict(checkpoint_G['state_dict'])
        age_optim_E.load_state_dict(checkpoint_E['optimizer'])
        age_optim_G.load_state_dict(checkpoint_G['optimizer'])


    enc_z = []
    enc_fake_z = []
    enc_rec_x = []
    gen_fake_z = []
    gen_rec_z = []

    age_E.train()
    age_G.train()

    for epoch in tqdm(range(NUM_EPOCH)):
        if epoch % DROP_LR == (DROP_LR - 1):
            LR /= 2
            for param_group in age_optim_E.param_groups:
                param_group['lr'] = LR

            for param_group in age_optim_G.param_groups:
                param_group['lr'] = LR

        for i, data in enumerate(trainloader, 0):
            input, label = data
            x.data.copy_(input)
            #batch_size = list(x.size())[0]
            # Update encoder
            loss_E = []
            age_optim_E.zero_grad()

            z = age_E(x)
            KL_z = KL_min(z)
            loss_E.append(KL_z)
            enc_z.append(KL_z)

            x_rec = age_G(z)
            x_rec_loss = REC_MU * l1_loss(x, x_rec)
            loss_E.append(x_rec_loss)
            enc_rec_x.append(x_rec_loss)

            z_sample.copy_(sampling(batch_size, Z_DIM))
            x_fake = age_G(z_sample).detach()
            z_fake = age_E(x_fake)
            KL_z_fake = KL_max(z_fake)
            loss_E.append(KL_z_fake)
            enc_fake_z.append(KL_z_fake)

            sum(loss_E).backward()
            age_optim_E.step()

            if i == 0:
                print('--------------------------------')
                print(f'Epoch: {epoch+1}, Batch: {i+1}')
                print(f'Encoder: KL z: {KL_z}, Rec x loss: {x_rec_loss}, KL fake z: {KL_z_fake}')

            # Update generator
            for g_i in range(G_UPDATES):
                loss_G = []
                age_optim_G.zero_grad()

                z_sample.copy_(sampling(batch_size, Z_DIM))
                x_fake = age_G(z_sample)
                z_fake = age_E(x_fake)
                KL_z_fake = KL_min(z_fake)
                loss_G.append(KL_z_fake)
                gen_fake_z.append(KL_z_fake)

                z_rec_loss = REC_LAMBDA * l2_loss(z_fake, z_sample)
                loss_G.append(z_rec_loss)
                gen_rec_z.append(z_rec_loss)

                sum(loss_G).backward()
                age_optim_G.step()
            if i == 0:
                print(f'Generator: KL fake z: {KL_z_fake}, Rec z loss: {z_rec_loss}')
                print('--------------------------------')

            # Clear GPU ??????
            if epoch == 0 and i == 0:
                torch.cuda.empty_cache()
            #torch.cuda.empty_cache()
            ########

        if epoch % save_model == (save_model - 1):
            age_im_gen(age_G, SAMPLE_BATCH, Z_DIM, device, f'{model_dir}/_img_{epoch}.png')
            state_E = {
                'epoch': epoch,
                'state_dict': age_E.state_dict(),
                'optimizer': age_optim_E.state_dict(),
            }
            state_G = {
                'epoch': epoch,
                'state_dict': age_G.state_dict(),
                'optimizer': age_optim_G.state_dict(),
            }
            torch.save(state_E, f"{model_dir}/encoder_{epoch}")
            torch.save(state_G, f"{model_dir}/generator_{epoch}")

    age_im_gen(age_G, SAMPLE_BATCH, Z_DIM, device, f'{model_dir}/_img_{epoch}.png')
    state_E = {
        'epoch': epoch,
        'state_dict': age_E.state_dict(),
        'optimizer': age_optim_E.state_dict(),
    }
    state_G = {
        'epoch': epoch,
        'state_dict': age_G.state_dict(),
        'optimizer': age_optim_G.state_dict(),
    }
    torch.save(state_E, f"{model_dir}/encoder_{epoch}")
    torch.save(state_G, f"{model_dir}/generator_{epoch}")

    plt.plot(enc_z, label='Encoder KL z')
    np.asarray(enc_z).dump(f"{plot_dir}/enc_z.dat")
    plt.plot(enc_fake_z, label='Encoder KL fake z')
    np.asarray(enc_fake_z).dump(f"{plot_dir}/enc_fake_z.dat")
    plt.legend()
    plt.savefig(f"{plot_dir}/encoder_KL")
    plt.close()

    plt.plot(gen_fake_z, label='Generator KL fake z')
    np.asarray(gen_fake_z).dump(f"{plot_dir}/gen_fake_z.dat")
    plt.legend()
    plt.savefig(f"{plot_dir}/generator_KL")
    plt.close()

    plt.plot(enc_rec_x, label='Encoder x recon')
    np.asarray(enc_rec_x).dump(f"{plot_dir}/enc_rec_x.dat")
    plt.legend()
    plt.savefig(f"{plot_dir}/encoder_rec")
    plt.close()

    plt.plot(gen_rec_z, label='Generator z recon')
    np.asarray(gen_rec_z).dump(f"{plot_dir}/gen_rec_z.dat")
    plt.legend()
    plt.savefig(f"{plot_dir}/generator_rec")
    plt.close()


