from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from AGE.age_networks import *
from AGE.loss_functions import *
from AGE.tools import *
import sys
import os

root_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
model_dir = os.path.join(root_dir, 'age_model_cifar')
plot_dir = os.path.join(root_dir, 'age_plot_cifar')

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    #trainset.data = trainset.data[np.where(np.array(trainset.targets)==1)] # Only cars
    #indice = list(range(0, 10000))
    # sampler=data.SubsetRandomSampler(indice)
    #trainset = torchvision.datasets.SVHN(root='.\data', transform=transform, download =True)
    trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
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

    cudnn.benchmark = True

    age_E = age_enc(z_dim=Z_DIM).cuda()
    age_G = age_gen(z_dim=Z_DIM).cuda()

    KL_min = KL_Loss_AGE(minimize=True).cuda()
    KL_max = KL_Loss_AGE(minimize=False).cuda()
    loss_l1 = nn.L1Loss().cuda()
    loss_l2 = nn.MSELoss().cuda()

    age_optim_E = optim.Adam(age_E.parameters(), lr=LR, betas=(0.5, 0.999))
    age_optim_G = optim.Adam(age_G.parameters(), lr=LR, betas=(0.5, 0.999))

    enc_z = []
    enc_fake_z = []
    enc_rec_x = []
    gen_fake_z = []
    gen_rec_z = []

    for epoch in tqdm(range(NUM_EPOCH)):
        if epoch % DROP_LR == (DROP_LR - 1):
            LR /= 2
            for param_group in age_optim_E.param_groups:
                param_group['lr'] = LR

            for param_group in age_optim_G.param_groups:
                param_group['lr'] = LR
            torch.save(age_E.state_dict(), f"{model_dir}/encoder_{epoch}")
            torch.save(age_G.state_dict(), f"{model_dir}/generator_{epoch}")
        for i, data in enumerate(trainloader, 0):
            #print(f'Epoch: {epoch+1}, Batch: i+1')
            #print('--------------------------------')
            x, label = data
            x = x.cuda()
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

            z_sample = sampling(batch_size, Z_DIM).cuda()
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

                z_sample = sampling(batch_size, Z_DIM).cuda()
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

                #print(f'Generator: KL fake z - {KL_z_fake}, Rec z loss - {z_rec_loss}')

    # plt.plot(enc_z, label='Encoder KL z')
    # plt.plot(enc_fake_z, label='Encoder KL fake z')
    # plt.legend()
    # plt.savefig()
    # #plt.show()
    # plt.plot(gen_fake_z, label='Generator KL fake z')
    # plt.legend()
    # plt.show()
    # plt.plot(enc_rec_x, label='Encoder x recon')
    # plt.legend()
    # plt.show()
    # plt.plot(gen_rec_z, label='Generator z recon')
    # plt.legend()
    # plt.show()

    # with torch.no_grad():
    #     for i in range(5):
    #         z_sample = sampling(1, Z_DIM).cuda()
    #         x_fake = age_G(z_sample).view(3, 32, 32).cpu()
    #         im_show(x_fake)
    #     recon_i = 0
    #     for i, data in enumerate(trainloader,0):
    #         x, label = data
    #         x = x[0].view(1, 3, 32, 32).cuda()
    #         z = age_E(x)
    #         x_rec = age_G(z).view(3, 32, 32).cpu()
    #         im_show(x.view(3, 32, 32).cpu())
    #         im_show(x_rec)
    #         recon_i += 1
    #         if recon_i == 5:
    #             break


