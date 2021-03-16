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

from config import opt

class CWRUDataset1D(data.Dataset):

    def __init__(self, filename, train=True):
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

    def __init__(self, root, train=True):
        '''
        获取所有图片的地址，并根据训练，测试划分数据（就不搞验证集了）
        :param root: 图片目录
        :param train: 是否为训练
        :param test: 是否为测试
        '''
        self.train_fraction = opt.train_fraction
        # 输出全部的图片
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # train: data/CWRU_data_2d/DE/gadf/0.35.png
        # test: test文件从train里面分出来的

        # 对图片的id进行排序 ['0', '35', 'png']
        imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
        imgs_num = len(imgs)

        # 训练数据集
        if train:
            self.imgs = imgs[:int(self.train_fraction * imgs_num)]
        else:
            self.imgs = imgs[int(self.train_fraction * imgs_num):]

        self.transforms = T.Compose([T.ToTensor()])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        如果是测试集，没有图片id
        """
        img_path = self.imgs[index]
        # self.imgs[index] == ./data/CWRU_data_2d/DE/gadf\97.62.png
        label = int(self.imgs[index].split('/')[-1].split('\\')[-1].split('.')[0])
        # 图片数据
        data = Image.open(img_path)
        # 这里需要将Image对象转换成tensor
        data = self.transforms(data)

        return data, label

    def __len__(self):
        return len(self.imgs)