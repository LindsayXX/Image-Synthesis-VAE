from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from loss_functions import *
from tools import *
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from constants import *

if __name__ == '__main__':
    cudnn = CUDA_BECHMARK
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
        trainloader, IMG_DIM, Z_DIM, model_dir, plot_dir = load_data('celebA', root=root, batch_size=INTRO_BATCH_SIZE,
                                                                     imgsz=128, num_worker=0)

    else:
        device = torch.device('cpu')
        print(f'Used device: {device}')
        ngpu = 0
        root_dir = 'D:\MY1\DPDS\project\DD2424-Projekt\data'
        trainloader, IMG_DIM, Z_DIM, model_dir, plot_dir = load_data('celebAtest', root=root_dir,
                                                                     batch_size=INTRO_BATCH_SIZE,
                                                                     imgsz=256, num_worker=0)

    # ------- build model -----------
    intro_E = Intro_enc(img_dim=IMG_DIM, z_dim=Z_DIM, ngpu=ngpu).to(device)
    intro_G = Intro_gen(z_dim=Z_DIM, ngpu=ngpu).to(device)

    # -------- load models if needed --------
    '''
    intro_E, intro_G = load_model(IMG_DIM, Z_DIM, ngpu, model_dir)
    '''

    optimizer_E = optim.Adam(intro_E.parameters(), lr=INTRO_LR, betas=(0.9, 0.999))
    optimizer_G = optim.Adam(intro_G.parameters(), lr=INTRO_LR, betas=(0.9, 0.999))

    intro_E.train()
    intro_G.train()

    x = torch.FloatTensor(INTRO_BATCH_SIZE, 3, IMG_DIM, IMG_DIM).to(device)
    z_p = torch.FloatTensor(INTRO_BATCH_SIZE, Z_DIM).to(device)
    x = Variable(x)
    z_p = Variable(z_p)

    KL_min = KL_Loss_Intro(minimize=True)
    KL_max = KL_Loss_Intro(minimize=False)
    # loss_l1 = nn.L1Loss()
    # loss_l2 = nn.MSELoss()

    enc_z = []
    enc_fake_z = []
    enc_rec_z = []
    rec_x = []
    gen_fake_z = []
    gen_rec_z = []

    for epoch in tqdm(range(START_EPOCH, NUM_EPOCH)):
        for i, data in enumerate(trainloader, 0):
            if epoch == START_EPOCH and i == 1 and LOAD_MODEL:
                print('Loading model!')
                checkpoint_E = torch.load(f"{model_dir}/encoder_{START_EPOCH - 1}")
                checkpoint_G = torch.load(f"{model_dir}/generator_{START_EPOCH - 1}")
                intro_E.load_state_dict(checkpoint_E['state_dict'])
                intro_G.load_state_dict(checkpoint_G['state_dict'])
                optimizer_E.load_state_dict(checkpoint_E['optimizer'])
                optimizer_G.load_state_dict(checkpoint_G['optimizer'])

            # print('Batch:',i)
            input, label = data
            x.data.copy_(input)

            # ----- update the encoder -------
            loss_E = []
            mean, logvar = intro_E(x)
            z = reparameterization(mean, logvar, ngpu, Z_DIM, device, INTRO_BATCH_SIZE)
            z_p.data.copy_(sampling(INTRO_BATCH_SIZE, Z_DIM, sphere=False, intro=True))

            x_r = intro_G(z)
            x_p = intro_G(z_p)
            L_ae = beta * l2_loss(x_r, x, age=False)
            loss_E.append(L_ae)
            rec_x.append(L_ae.cpu().data)
            mean_r, logvar_r = intro_E(x_r.detach())
            # z_r = reparameterization(mean_r, logvar_r)
            mean_pp, logvar_pp = intro_E(x_p.detach())
            # z_pp = reparameterization(mean_pp, logvar_pp)
            KL_z = KL_min(mean, logvar)
            loss_E.append(KL_z)
            enc_z.append(KL_z.cpu().data)
            # max(0, x) = ReLu(x)
            KL_rec_z = KL_max(mean_r, logvar_r)
            KL_fake_z = KL_max(mean_pp, logvar_pp)
            L_adv_E = (F.relu(M + KL_rec_z) + F.relu(M + KL_fake_z)).mul(alpha)
            loss_E.append(L_adv_E)
            enc_rec_z.append(KL_rec_z.cpu().data)
            enc_fake_z.append(KL_fake_z.cpu().data)

            optimizer_E.zero_grad()
            optimizer_G.zero_grad()
            sum(loss_E).backward(
                retain_graph=True)  # keep the variable after doing back`ward, for the backprop of Generator
            optimizer_E.step()

            # ----- update the generator/decoder -------
            loss_G = []
            mean_r_g, logvar_r_g = intro_E(x_r)
            mean_pp_g, logvar_pp_g = intro_E(x_p)
            KL_gen_rec = KL_min(mean_r_g, logvar_r_g)
            KL_gen_fake = KL_min(mean_pp_g, logvar_pp_g)
            L_adv_G = alpha * (KL_gen_rec + KL_gen_fake)
            loss_G.append(L_adv_G)  # + L_ae?
            gen_rec_z.append(KL_gen_rec.cpu().data)
            gen_fake_z.append(KL_gen_fake.cpu().data)

            sum(loss_G).backward()
            optimizer_G.step()

            if i % PRINT_STATS == (PRINT_STATS - 1):
                print('--------------------------')
                print(f'Epoch: {epoch}, Batch: {i}')
                print('Encoder:')
                print(f'KL Z: {KL_z}')
                print(f'Rec x: {L_ae}')
                print(f'KL fake z: {KL_fake_z}, KL rec z: {KL_rec_z}')
                print('Generator:')
                print(f'KL fake z: {KL_gen_fake}, KL rec z: {KL_gen_rec}')
                print('--------------------------')
        # --------- save model in every {save_model} epoches ----------
        if epoch % save_model == (save_model - 1):
            vae_im_gen(intro_G, SAMPLE_BATCH, Z_DIM, device, f'{model_dir}/_img_{epoch}.png')
            vae_im_rec(intro_E, intro_G, x, f'{model_dir}/rec_{epoch}.png', ngpu, Z_DIM=Z_DIM, data='celebA')
            state_E = {
                'epoch': epoch,
                'state_dict': intro_E.state_dict(),
                'optimizer': optimizer_E.state_dict(),
            }
            state_G = {
                'epoch': epoch,
                'state_dict': intro_G.state_dict(),
                'optimizer': optimizer_G.state_dict(),
            }
            torch.save(state_E, f"{model_dir}/encoder_{epoch}")
            torch.save(state_G, f"{model_dir}/generator_{epoch}")

    # ---------- save model -------------
    vae_im_gen(intro_G, SAMPLE_BATCH, Z_DIM, device, f'{model_dir}/_img_{epoch}.png')
    vae_im_rec(intro_E, intro_G, x, f'{model_dir}/rec_{epoch}.png', ngpu, Z_DIM=Z_DIM, data='celebA')
    state_E = {
        'epoch': epoch,
        'state_dict': intro_E.state_dict(),
        'optimizer': optimizer_E.state_dict(),
    }
    state_G = {
        'epoch': epoch,
        'state_dict': intro_G.state_dict(),
        'optimizer': optimizer_G.state_dict(),
    }
    torch.save(state_E, f"{model_dir}/encoder_{epoch}")
    torch.save(state_G, f"{model_dir}/generator_{epoch}")

    plt.plot(enc_z, label='Encoder KL z')
    np.asarray(enc_z).dump(f"{plot_dir}/enc_z.dat")
    plt.plot(enc_fake_z, label='Encoder KL fake z')
    np.asarray(enc_fake_z).dump(f"{plot_dir}/enc_fake_z.dat")
    plt.plot(enc_rec_z, label='Encoder KL rec z')
    np.asarray(enc_rec_z).dump(f"{plot_dir}/enc_rec_z.dat")
    plt.legend()
    plt.savefig(f"{plot_dir}/encoder_KL")
    plt.close()

    plt.plot(gen_fake_z, label='Generator KL fake z')
    np.asarray(gen_fake_z).dump(f"{plot_dir}/gen_fake_z.dat")
    plt.plot(gen_rec_z, label='Generator KL rec z')
    np.asarray(gen_rec_z).dump(f"{plot_dir}/gen_rec_z.dat")
    plt.legend()
    plt.savefig(f"{plot_dir}/generator_KL")
    plt.close()

    plt.plot(rec_x, label='x recon')
    np.asarray(rec_x).dump(f"{plot_dir}/rec_x.dat")
    plt.legend()
    plt.savefig(f"{plot_dir}/rec_x")
    plt.close()
