from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from IntroVAE.introvae_networks import *
from .loss_functions import *
from tools import *
import os
import constants as c
if __name__ == '__main__':
    cudnn.benchmark = c.CUDA_BECHMARK

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f'Cuda available: {torch.cuda.is_available()}')
        print(f'Current device number: {torch.cuda.current_device()}')
        print(f'Current device: {torch.cuda.device(torch.cuda.current_device())}')
        print(f'Number of GPUs: {torch.cuda.device_count()}')
        print(f'Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        print(f'Used device: {device}')
        ngpu = torch.cuda.device_count()
        root = os.path.abspath(os.path.dirname(__file__)) + '/../data/'
        trainloader, IMG_DIM, Z_DIM, model_dir, plot_dir = load_data('celebA', root=root, batch_size=c.AGE_BATCH_SIZE,
                                                                     imgsz=128, num_worker=0)

    else:
        device = torch.device('cpu')
        print(f'Used device: {device}')
        ngpu = 0
        root_dir = 'D:\MY1\DPDS\project\DD2424-Projekt\data'
        trainloader, IMG_DIM, Z_DIM, model_dir, plot_dir = load_data('celebAtest', root=root_dir, batch_size=c.AGE_BATCH_SIZE,
                                                                     imgsz=256, num_worker=4)

    # ------- build model -----------
    intro_E = Intro_enc(img_dim=IMG_DIM, z_dim=Z_DIM, ngpu=ngpu).to(device)
    intro_G = Intro_gen(z_dim=Z_DIM, ngpu=ngpu).to(device)

    intro_E.train()
    intro_G.train()

    # -------- load models if needed --------

    optimizer_E = optim.Adam(intro_E.parameters(), lr=c.LR, betas=(0.9, 0.999))
    optimizer_G = optim.Adam(intro_G.parameters(), lr=c.LR, betas=(0.9, 0.999))

    x = torch.FloatTensor(c.VAE_BATCH_SIZE, 3, c.VAE_IM_DIM, c.VAE_IM_DIM).to(device)
    z_p = torch.FloatTensor(c.VAE_BATCH_SIZE, c.VAE_Z_DIM).to(device)
    x = Variable(x)
    z_p = Variable(z_p)

    KL_min = KL_Loss_Intro(minimize=True)
    # loss_l1 = nn.L1Loss()
    # loss_l2 = nn.MSELoss()
    enc_z = []
    rec_x = []

    for epoch in tqdm(range(c.VAE_NUM_EPOCH)):
        # --------- save model in every 100 epoches ----------
        for i, data in enumerate(trainloader, 0):
            # print('Batch:',i)
            input, label = data
            x.data.copy_(input)

            # ----- update the encoder -------
            loss_vae = []
            optimizer_E.zero_grad()
            optimizer_G.zero_grad()
            mean, logvar = intro_E(x)
            z = reparameterization(mean, logvar, ngpu)

            x_r = intro_G(z)
            L_ae = l2_loss(x_r, x, age=False)

            loss_vae.append(L_ae)
            KL_z = KL_min(mean, logvar)
            loss_vae.append(KL_z)
            rec_x.append(L_ae)
            enc_z.append(KL_z)

            sum(loss_vae).backward()
            optimizer_E.step()
            optimizer_G.step()

        if epoch % c.AGE_BATCH_SIZE == (c.VAE_EPOCHS_PER_SAVE - 1):
            vae_im_gen(intro_G, c.AGE_BATCH_SIZE, Z_DIM, device, f'{model_dir}/_img_{epoch}.png')
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
    vae_im_gen(intro_G, c.VAE_SAMPLE_BATCH, Z_DIM, device, f'{model_dir}/_img_{epoch}.png')
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
    plt.legend()
    plt.savefig(f"{plot_dir}/encoder_KL")
    plt.close()

    plt.plot(rec_x, label='x recon')
    np.asarray(rec_x).dump(f"{plot_dir}/rec_x.dat")
    plt.legend()
    plt.savefig(f"{plot_dir}/rec_x")
    plt.close()
