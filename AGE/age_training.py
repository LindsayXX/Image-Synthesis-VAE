from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from AGE.age_networks import *
from AGE.loss_functions import *
from AGE.tools import *
import sys
import os

root_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
if not os.path.exists('age_model_cifar'):
    os.mkdir('age_model_cifar')
if not os.path.exists('age_plot_cifar'):
    os.mkdir('age_plot_cifar')
model_dir = os.path.join(root_dir, 'age_model_cifar')
plot_dir = os.path.join(root_dir, 'age_plot_cifar')

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset.data = trainset.data[np.where(np.array(trainset.targets)==1)] # Only cars
    #indice = list(range(0, 10000))
    # sampler=data.SubsetRandomSampler(indice)
    #trainset = torchvision.datasets.SVHN(root='.\data', transform=transform, download =True)
    trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #testset_sub = torch.utils.data.SubsetRandomSampler(indice)
    #testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    NUM_EPOCH = 150
    REC_LAMBDA = 1000
    REC_MU = 10
    Z_DIM = 128
    DROP_LR = 40
    LR = 0.0002

    age_E = age_enc(z_dim=Z_DIM).cuda()
    age_G = age_gen(z_dim=Z_DIM).cuda()
    age_E.apply(weights_init)
    age_G.apply(weights_init)
    age_E.train()
    age_G.train()

    x = torch.FloatTensor(64, 3, 32, 32).cuda()
    z_sample = torch.FloatTensor(64, Z_DIM, 1, 1).cuda()
    x = Variable(x)
    z_sample = Variable(z_sample)

    KL_min = KL_Loss_AGE(minimize=True)
    KL_max = KL_Loss_AGE(minimize=False)
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()

    age_optim_E = optim.Adam(age_E.parameters(), lr=LR, betas=(0.5, 0.999))
    age_optim_G = optim.Adam(age_G.parameters(), lr=LR, betas=(0.5, 0.999))

    enc_z = []
    enc_fake_z = []
    enc_rec_x = []
    gen_fake_z = []
    gen_rec_z = []

    cudnn.benchmark = True

    for epoch in tqdm(range(NUM_EPOCH)):
        if epoch % DROP_LR == (DROP_LR - 1):
            LR /= 2
            for param_group in age_optim_E.param_groups:
                param_group['lr'] = LR

            for param_group in age_optim_G.param_groups:
                param_group['lr'] = LR

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
        for i, data in enumerate(trainloader, 0):
            #print(f'Epoch: {epoch+1}, Batch: i+1')
            #print('--------------------------------')
            input, label = data
            x.data.copy_(input)
            batch_size = list(x.size())[0]
            # Update encoder
            loss_E = []
            age_optim_E.zero_grad()

            z = age_E(x)
            KL_z = KL_min(z)
            loss_E.append(KL_z)
            enc_z.append(KL_z)

            x_rec = age_G(z)
            x_rec_loss = REC_MU*loss_l1(x, x_rec)
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

            #print(f'Encoder: KL z - {KL_z}, Rec x loss - {x_rec_loss}, - KL fake z - {KL_z_fake}')

            # Update generator
            for g_i in range(2):
                loss_G = []
                age_optim_G.zero_grad()

                z_sample.copy_(sampling(batch_size, Z_DIM))
                x_fake = age_G(z_sample)
                z_fake = age_E(x_fake)
                KL_z_fake = KL_min(z_fake)
                loss_G.append(KL_z_fake)
                gen_fake_z.append(KL_z_fake)

                z_rec_loss = REC_LAMBDA * loss_l2(z_fake, z_sample)
                loss_G.append(z_rec_loss)
                gen_rec_z.append(z_rec_loss)

                sum(loss_G).backward()
                age_optim_G.step()

                # print(f'Generator: KL fake z - {KL_z_fake}, Rec z loss - {z_rec_loss}')

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
    plt.plot(enc_fake_z, label='Encoder KL fake z')
    plt.legend()
    plt.savefig(f"{plot_dir}/encoder_KL")
    plt.close()

    plt.plot(gen_fake_z, label='Generator KL fake z')
    plt.legend()
    plt.savefig(f"{plot_dir}/generator_KL")
    plt.close()

    plt.plot(enc_rec_x, label='Encoder x recon')
    plt.legend()
    plt.savefig(f"{plot_dir}/encoder_rec")
    plt.close()

    plt.plot(gen_rec_z, label='Generator z recon')
    plt.legend()
    plt.savefig(f"{plot_dir}/generator_rec")
    plt.close()


