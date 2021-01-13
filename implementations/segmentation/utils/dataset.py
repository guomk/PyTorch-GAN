# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:55
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : dataset.py
"""

"""
import os
import os.path as osp
import logging

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, img_size):
        self.imgs_dir   = imgs_dir
        self.masks_dir  = masks_dir
        self.img_size   = img_size

        # assert 0 < img_size <= 1, 'Scale must be between 0 and 1'

        self.img_names = os.listdir(imgs_dir)
        logging.info(f'Creating dataset with {len(self.img_names)} examples')

    def __len__(self):
        return len(self.img_names)

    @classmethod
    def preprocess(cls, pil_img, img_size):
        w, h = pil_img.size
        assert w == h, 'Can only use square image'
#         print('before resizing', pil_img.size)
        # newW, newH = img_size, img_size
        assert img_size > 0, 'New image size cannot be less than 0'
        pil_img = pil_img.resize((img_size, img_size))
#         print('after resizing', pil_img.size)

        img_nd = np.array(pil_img)
#         print('np array', img_nd.shape)
        if len(img_nd.shape) == 2:
            # mask target image
            img_nd = np.expand_dims(img_nd, axis=2)
        else:
            # grayscale input image
            # scale between 0 and 1
            img_nd = img_nd / 255
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans.astype(float)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, img_name)
        mask_path = osp.join(self.masks_dir, img_name)

        img = Image.open(img_path).convert("L")
#         print('before', img.size)
        mask = Image.open(mask_path).convert("L")

        assert img.size == mask.size, \
            f'Image and mask {img_name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
#         print('after', img.shape)
        mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
