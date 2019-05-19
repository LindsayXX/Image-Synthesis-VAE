import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data as data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import os
from matplotlib import pyplot as plt

this_root = os.path.abspath(os.path.dirname(__file__))


def show(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """

    :param imgs: Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    :param cuda: Boolean whether or not to run on GPU
    :param batch_size: batch size for feeding into Inception v3
    :param splits: number of splits
    :return: The mean and the variance of the Inception score
    """

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    # indice = list(range(0, 20*batch_size))
    # sampler = data.SubsetRandomSampler(indice)
    #
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), align_corners=False, mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)

        return F.softmax(input=x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))
    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)
        print(f'{batch_size + i * batch_size} evaluated samples')

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def load_fake_data(path_to_fake):
    """
    Temp function to test the inception score
    :param path_to_fake: Path to a dataset
    :return: A data loader object of the dataset
    """

    if os.path.isdir(path_to_fake):
        dataset = []
        for file in os.listdir(path_to_fake):
            filename = this_root + '/tensors/' + file
            data_file = torch.load(filename)
            dataset.append(data_file)

        return torch.cat(dataset)

    else:
        return torch.load(os.path.join(path_to_fake))


if __name__ == '__main__':
    pass
