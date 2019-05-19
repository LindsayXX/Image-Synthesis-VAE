import os
import torch
import torch.utils.data as data
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms


this_root = os.path.abspath(os.path.dirname(__file__))


def load_image(file_path, input_height=128, input_width=None, output_height=128, output_width=None,
               crop_height=None, crop_width=None, is_random_crop=True, is_mirror=False, is_gray=False):
    if input_width is None:
        input_width = input_height
    if output_width is None:
        output_width = output_height
    if crop_width is None:
        crop_width = crop_height

    img = Image.open(file_path)
    if is_gray is False and img.mode is not 'RGB':
        img = img.convert('RGB')
    if is_gray and img.mode is not 'L':
        img = img.convert('L')

    if is_mirror and random.randint(0, 1) is 0:
        img = ImageOps.mirror(img)

    if input_height is not None:
        img = img.resize((input_width, input_height), Image.BICUBIC)

    if crop_height is not None:
        [w, h] = img.size
        if is_random_crop:
            cx1 = random.randint(0, w - crop_width)
            cx2 = w - crop_width - cx1
            cy1 = random.randint(0, h - crop_height)
            cy2 = h - crop_height - cy1
        else:
            cx2 = cx1 = int(round((w - crop_width) / 2.))
            cy2 = cy1 = int(round((h - crop_height) / 2.))
        img = ImageOps.crop(img, (cx1, cy1, cx2, cy2))

    img = img.resize((output_height, output_width), Image.BICUBIC)
    return img


def load_fake_image(img, input_height, input_width, output_height, output_width):
    fake_image = torch.load(img)
    return fake_image


def get_list_filenames(root_path):
    list = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if not file.endswith(".jpg"):
                continue
            path = os.path.join(root, file).replace(root_path, '')
            list.append(path)


class Dataset(data.Dataset):
    def __init__(self, root_path, filename='1000_fake_tensor_cifar_10', dataset_type='celeba', input_height=128,
                 crop_height=None, crop_width=None, is_random_crop=False, is_mirror=True,
                 is_gray=False):
        """

        :param root_path: Path to the directory of the dataset
        :param filename: Name of the file
        :param dataset_type: Which dataset we are referring to
        :param input_height: Height of the image. Default set to 128
        :param crop_height:
        :param crop_width:
        :param is_random_crop:
        :param is_mirror:
        :param is_gray:
        """
        super(Dataset, self).__init__()
        self.dataset_type = dataset_type
        self.root_path = root_path
        self.input_height = input_height
        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.filename = filename
        self.is_gray = is_gray

        if dataset_type is 'celeba':
            self.image_filenames = get_list_filenames(root_path)

            self.input_transform = transforms.Compose([

                transforms.Resize([self.input_height, self.input_height]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

            # db = datasets.ImageFolder(root, transform=transform)
            indice = list(range(0, 5000))
            try_sampler = data.SubsetRandomSampler(indice)

    def __getitem__(self, index):
        if self.dataset_type is 'celeba':
            img = load_image(join(self.root_path, self.image_filenames[index]),
                             self.input_height, self.input_width, self.output_height, self.output_width,
                             self.crop_height, self.crop_width, self.is_random_crop, self.is_mirror, self.is_gray)

            img = self.input_transform(img)

    def __len__(self):
        return len(self.image_filenames)


if __name__ == '__main__':
    trainset = Dataset(this_root, dataset_type='fake_generated')
    trainloader = data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    # transform = transforms.Compose([ transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    # torch_file = os.path.join(this_root, '4_fake_tensor_cifar_10')
    #
    # fake = torch.load(torch_file)

    # print(len(fake))
