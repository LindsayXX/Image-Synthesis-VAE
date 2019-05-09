import os

import torch
import torch.utils.data as data
from os.path import join
from PIL import Image, ImageOps
import random
import torchvision.transforms as transforms


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

class DataLoader(data.Dataset):
    def __init__(self, image_list, root_path=None, dataset_type='celeba',
                 input_height=128, input_width=None,
                 crop_height=None, crop_width=None, is_random_crop=False, is_mirror=True, is_gray=False):

        super(DataLoader, self).__init__()
        self.dataset_type = dataset_type
        self.image_filenames = image_list
        self.root_path = root_path

        self.is_random_crop = is_random_crop
        self.is_mirror = is_mirror
        self.crop_height = crop_height
        self.crop_width = crop_width

        self.is_gray = is_gray

        if dataset_type is 'celeba':
            self.input_transform = transforms.Compose([
                transforms.Resize([input_width, input_height]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            self.input_height = 128
            self.input_width = 128
        elif dataset_type is 'fake_generated':

            self.input_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            self.input_height = 32
            self.input_width = 32

        self.output_height = self.input_height
        self.output_width = self.input_width

    def __getitem__(self, index):
        if self.dataset_type is 'celeba':
            img = load_image(join(self.root_path, self.image_filenames[index]),
                             self.input_height, self.input_width, self.output_height, self.output_width,
                             self.crop_height, self.crop_width, self.is_random_crop, self.is_mirror, self.is_gray)

            img = self.input_transform(img)

        if self.dataset_type is 'fake_generated':
            img = load_fake_image(self.image_filenames[index],
                                  self.input_height,
                                  self.input_width,
                                  self.output_height,
                                  self.output_width)

            img = self.input_transform(img)

        return img

    def __len__(self):
        return len(self.image_filenames)


def load_data(root_dir):

    list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".png"):
                continue
            path = os.path.join(root, file).replace(root_dir, '')
            list.append(path)




    db = DataLoader(list, root_dir, dataset_type='celeba')
    return db
