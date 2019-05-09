from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from IntroVAE.introvae_networks import *
from AGE.loss_functions import *
from AGE.tools import *
import os
import sys

def load_model():
    # TODO code for load the models
    return 0

def reparameterization(mean, logvar, ngpu=1):
    # TODO z = mu + sigma.mul(eps)
    std = logvar.mul(0.5).exp_()
    z = eps.mul(std).add_(mean)

    return z.unsqueeze_(-2)


def load_data(dataset='celebA', root='.\data', batch_size=64, imgsz=128, num_worker=4):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    root_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
    if not os.path.exists('intro_model_{}'.format(dataset)):
        os.mkdir('Intro_model_{}'.format(dataset))
    if not os.path.exists('Intro_plot_{}'.format(dataset)):
        os.mkdir('Intro_plot_{}'.format(dataset))
    model_dir = os.path.join(root_dir, 'Intro_model_{}'.format(dataset))
    plot_dir = os.path.join(root_dir, 'Intro_plot_{}'.format(dataset))

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
        indice = list(range(0, 10))
        try_sampler = data.SubsetRandomSampler(indice)
        trainloader = data.DataLoader(db, batch_size=batch_size, shuffle=False, num_workers=num_worker, sampler=try_sampler)
        #trainloader = data.DataLoader(db, batch_size=batch_size, shuffle=True, num_workers=num_worker)

        if imgsz == 128:
            z_dim = 256
        elif imgsz == 256:
            z_dim = 512

    return trainloader, imgsz, z_dim


if __name__ == '__main__':
    """
    for 256*256 imgae size
    --m_plus=120 --weight_rec=0.05 --weight_kl=1.0  --weight_neg=0.5 --num_vae=0 
    --trainsize=29000 --test_iter=1000 --save_iter=1 --start_epoch=0  --batchSize=16 
    --nrow=8 --lr_e=0.0002 --lr_g=0.0002   --cuda  --nEpochs=500
    or
    --m_plus=1000 --weight_rec=1.0  --num_vae=10
    """
    NUM_EPOCH = 2 #500
    LR = 0.0002
    weight_rec = 0.05
    batch_size = 2 #16
    #IMG_DIM = 128#256
    #Z_DIM = 256#512
    alpha = 0.25
    beta = 0.05
    M = 120
    '''
    IMG_DIM = 128
    Z_DIM = 256
    alpha = 0.25
    beta = 0.5
    M = 110
    '''

    cudnn.benchmark = True

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f'Cuda available: {torch.cuda.is_available()}')
        print(f'Current device number: {torch.cuda.current_device()}')
        print(f'Current device: {torch.cuda.device(torch.cuda.current_device())}')
        print(f'Number of GPUs: {torch.cuda.device_count()}')
        print(f'Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        print(f'Used device: {device}')
        ngpu = torch.cuda.device_count()

    else:
        device = torch.device('cpu')
        print(f'Used device: {device}')
        ngpu = 0


    root_dir = 'D:\MY1\DPDS\project\DD2424-Projekt\data'
    trainloader, IMG_DIM, Z_DIM = load_data('celebA', root=root_dir, batch_size=batch_size, imgsz=128, num_worker=4)

    # ------- build model -----------
    intro_E = Intro_enc(img_dim=IMG_DIM, z_dim=Z_DIM, ngpu=ngpu).to(device)
    intro_G = Intro_gen(z_dim=Z_DIM, ngpu=ngpu).to(device)
    intro_E.apply(weights_init)
    intro_G.apply(weights_init)
    intro_E.train()
    intro_G.train()

    # -------- load models if needed --------
    '''
    intro_E = load_model(intro_E)
    intro_G = load_model(intro_G)
    '''

    optimizer_E = optim.Adam(intro_E.parameters(), lr=LR, betas=(0.9, 0.999))
    optimizer_G = optim.Adam(intro_G.parameters(), lr=LR, betas=(0.9, 0.999))

    x = torch.FloatTensor(batch_size, 3, IMG_DIM, IMG_DIM).to(device)
    z_p = torch.FloatTensor(batch_size, Z_DIM, 1, 1).to(device)
    z = torch.FloatTensor(batch_size, Z_DIM, 1, 1).to(device)
    eps = torch.FloatTensor(batch_size, Z_DIM).normal_().to(device)
    x = Variable(x)
    z_p = Variable(z_p)
    z = Variable(z)
    eps = Variable(eps)


    KL_min = KL_Loss_Intro(minimize=True)
    KL_max = KL_Loss_Intro(minimize=False)
    # loss_l1 = nn.L1Loss()
    # loss_l2 = nn.MSELoss()

    for epoch in tqdm(range(NUM_EPOCH)):
        for i, data in enumerate(trainloader, 0):
            print('Batch:',i)
            input, label = data
            x.data.copy_(input)

            # ----- update the encoder -------
            loss_E = []
            optimizer_E.zero_grad()
            mean, logvar = intro_E(x)
            z = reparameterization(mean, logvar, ngpu)
            z_p = sampling(batch_size, Z_DIM, sphere=False)
            x_r = intro_G(z)
            x_p = intro_G(z_p)
            L_ae = beta * l2_loss(x_r, x, age=False)
            loss_E.append(L_ae)
            mean_r, logvar_r = intro_E(x_r.detach())
            #z_r = reparameterization(mean_r, logvar_r)
            mean_pp, logvar_pp = intro_E(x_p.detach())
            #z_pp = reparameterization(mean_pp, logvar_pp)
            loss_E.append(KL_max(mean, logvar))
            # max(0, x) = ReLu(x)
            L_adv_E = (F.relu(M - KL_max(mean_r, logvar_r)) + F.relu(M - KL_max(mean_pp, logvar_pp))).mul(alpha)
            loss_E.append(L_adv_E)

            sum(loss_E).backward(retain_graph=True) # keep the variable after doing back`ward, for the backprop of Generator
            optimizer_E.step()

            # ----- update the generator/decoder -------
            loss_G = []
            optimizer_G.zero_grad()
            mean_r_g, logvar_r_g = intro_E(x_r)
            mean_pp_g, logvar_pp_g = intro_E(x_p)
            L_adv_G = alpha * (KL_min(mean_r_g, logvar_r_g) + KL_min(mean_pp_g, logvar_pp_g))
            loss_G.append(L_adv_G + beta * L_ae)

            sum(loss_G).backward()
            optimizer_G.step()



    # TODO: visualize some example and test
