#!/usr/bin/env python
# @Time    : 2021/3/8 15:40
# @Author  : wb
# @File    : dataset.py

'''
pytorch 读取数据
'''

import h5py
import os
from torchvision import transforms as T
from PIL import Image

import torch.utils.data as data

class CWRUDataset(data.Dataset):

    def __init__(self, filename, train):
        '''
        pytorch读取训练数据
        :param filename: 数据集文件，这边是h5py文件
        :param train: 是否为训练，还是测试
        '''
        f = h5py.File(filename, 'r')
        if train:
            self.X = f['X_train'][:]
            self.y = f['y_train'][:]
        else:
            self.X = f['X_test'][:]
            self.y = f['y_test'][:]

    def __getitem__(self, idx):
        '''
        返回一条数据
        :param idx:
        :return:
        '''
        return self.X[idx], self.y[idx]

    def __len__(self):
        '''
        数据长度
        :return:
        '''
        return self.X.shape[0]

class CWRUDataset2D(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomReSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)