import os
import glob
import numpy as np
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from abslog import abslog


def resize_long_side(img, target_length = 256):
    long_side_length = target_length
    oldh, oldw = img.shape[1], img.shape[2]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    newh, neww = int(newh + 0.5), int(neww + 0.5)
    target_size = (newh, neww)
    # resize transform instance
    resize_trans = transforms.Resize(size=target_size, antialias=True)
    img_resized = resize_trans(img)
    return img_resized


def pad_image(img, target_size = 256):
    h, w = img.shape[1], img.shape[2]
    pad_l = int((target_size - w) / 2)
    pad_r = int(target_size - w - pad_l)
    pad_t = int((target_size - h) / 2)
    pad_b = int(target_size - h - pad_t)
    pad_size = (pad_l, pad_t, pad_r, pad_b)
    pad_trans = transforms.Pad(pad_size)
    img_padded = pad_trans(img)
    return img_padded


def img_aug(img):
    # random augmentation
    # apply random horizontal flip
    if torch.rand(1) > 0.5:
        img = TF.hflip(img)
    # apply random vertical flip
    if torch.rand(1) > 0.5:
        img = TF.vflip(img)
    # apply random rotation
    if torch.rand(1) > 0.5:
        angle = transforms.RandomRotation.get_params(degrees=(-180, 180))
        img = TF.rotate(img, angle)
    return img


def value_norm(img):
    lower_bound, upper_bound = np.percentile(img[img > 0], 1.0), np.percentile(img[img > 0], 99.0)
    img_pre = np.clip(img, lower_bound, upper_bound)
    img_pre = ((img_pre - np.min(img_pre)) / (np.max(img_pre) - np.min(img_pre)) * 1.0)
    # img to tensor
    img_pre = np.squeeze(img_pre)
    img_3c = np.expand_dims(img_pre, axis=0)
    img_3c = torch.from_numpy(img_3c)
    return img_3c


def preprocess(img, image_size):
    # reshape the image
    img = resize_long_side(img, image_size)
    # pad the image
    img = pad_image(img, image_size)
    # return the image
    return img


def load_data(npy_path, image_size=256, aug=True):
    img = np.load(npy_path)
    img = value_norm(img)
    img = preprocess(img.to(torch.float32), image_size)
    # augment flips and rotatio
    if aug:
        img = img_aug(img)
    # then create a edge image
    struct = abslog(img)
    return img, struct


class UNCDataset(torch.utils.data.Dataset):
    def __init__(self, config, training):
        super(UNCDataset).__init__()
        # data parameters
        self.image_size = config.vqgan.image_size
        self.train_domain = config.dataloader.train_domain
        if training:
            glob_str = os.path.join(config.dataloader.data_root_dir, 'train' + config.dataloader.train_domain + '/*.npy')
        else:
            glob_str = os.path.join(config.dataloader.data_root_dir, 'test' + config.dataloader.test_domain + '/*.npy')
        self.npy_list = glob.glob(glob_str)
        random.shuffle(self.npy_list)
        # limit the samples number in one epoch
        total_len = int( config.dataloader.batch_size * config.dataloader.batch_num)
        self.npy_list = self.npy_list[:total_len]
    
    def __len__(self):
        return len(self.npy_list)
    
    def __getitem__(self, idx):
        now_npy_path = self.npy_list[idx]
        img, struct = load_data(now_npy_path, image_size=self.image_size)
        return img.to(torch.float32), struct.to(torch.float32)


def UNCDataloader(config, training=True):
    dataloader = UNCDataset(config, training)
    return torch.utils.data.DataLoader(
        dataloader,
        batch_size=config.dataloader.batch_size,
        collate_fn=None,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        pin_memory=True,
        drop_last=False)


if __name__ == '__main__':
    from utils import get_config
    config = get_config('./configs/unc-config.yaml')
    config.dataloader.num_workers = 1
    config.dataloader.batch_size = 3
    dataloader = UNCDataloader(config)
    count = 0
    for (img, struct) in dataloader:
        print(img.shape)
        print(struct.shape)
        count += 1
        if count > 10:
            break
