from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from .introvae_networks import *
from AGE.loss_functions import *
from AGE.tools import *
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

root_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
if not os.path.exists('intro_model_celebA'):
    os.mkdir('intro_model_celebA')
if not os.path.exists('intro_plot_celebA'):
    os.mkdir('intro_plot_celebA')
model_dir = os.path.join(root_dir, 'intro_model_celebA')
plot_dir = os.path.join(root_dir, 'intro_plot_celebA')

def readdata():
    transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def load_model():
    # TODO code for load the models
    return 0

def reparameterization(mean, logvar):
    # TODO z = mu + sigma.mul(eps)
    return 0


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
    M = 120
    LR = 0.0002
    weight_rec = 0.05
    batch_size = 16
    Z_DIM = 512
    ngpu = 1
    IMG_DIM = 256

    cudnn.benchmark = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Cuda available: {torch.cuda.is_available()}')
    print(f'Current device number: {torch.cuda.current_device()}')
    print(f'Current device: {torch.cuda.device(torch.cuda.current_device())}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    print(f'Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    print(f'Used device: {device}')

    # TODO: read data
    # -------------------------- load data ------------------------------------
    readdata()

    # ------- build model -----------
    intro_E = Intro_enc(img_dim=IMG_DIM, z_dim=Z_DIM, ngpu=ngpu).to(device)
    intro_G = Intro_gen(z_dim=Z_DIM, ngpu=ngpu).to(device)
    intro_E.apply(weights_init)
    intro_G.apply(weights_init)
    intro_E.train()
    intro_G.train()

    # load models if needed
    '''
    intro_E = load_model(intro_E)
    intro_G = load_model(intro_G)
    '''

    optimizer_E = optim.Adam(intro_E.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(intro_G.parameters(), lr=LR, betas=(0.5, 0.999))

    x = torch.FloatTensor(batch_size, 3, IMG_DIM, IMG_DIM).to(device)
    z_sample = torch.FloatTensor(batch_size, Z_DIM, 1, 1).to(device)
    x = Variable(x)
    z_sample = Variable(z_sample)

    KL_min = KL_Loss_Intro(minimize=True)
    KL_max = KL_Loss_Intro(minimize=False)
    # loss_l1 = nn.L1Loss()
    # loss_l2 = nn.MSELoss()

    for epoch in tqdm(range(NUM_EPOCH)):
        for i, data in enumerate(trainloader, 0):
            x.data.copy_(input)

            mean, logvar = intro_E(x)
            z = reparameterization(mean, logvar)
            z_p = sampling(batch_size, Z_DIM, sphere=False)






    # TODO: visualize some example and test
