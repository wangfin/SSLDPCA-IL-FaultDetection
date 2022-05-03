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
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

from config import opt

class CWRUDataset1D(data.Dataset):

    def __init__(self, filename, train=True):
        '''
        pytorch读取训练数据
        :param filename: 数据集文件，这边是h5py文件
        :param train: 是否为训练，还是测试
        '''
        f = h5py.File(filename, 'r')
        # 数据，取值，可以用f['data'].value，不过包自己推荐使用f['data'][()]这种方式
        data = f['data'][()]
        # 标签
        label = f['label'][()]
        # 每个类别的数据块数量
        data_num = f['data_num'][()]

        print(label)

        # 各个类别的数据
        category_data = []
        # 各个类别的标签
        category_label = []

        # 手动拆分下数据集
        # 把每个类别的数据切分出来，就是根据每个类别数据块的数量将数据拆分过来
        point = 0
        for i in range(len(data_num)):
            data_ = data[point:point + data_num[i]]
            label_ = label[point:point + data_num[i]]

            category_data.append(data_)
            category_label.append(label_)

            point = point + data_num[i]

        # 训练集与测试集
        train_X = np.empty(shape=(1, 400))
        train_y = np.empty(shape=(1,))
        test_X = np.empty(shape=(1, 400))
        test_y = np.empty(shape=(1,))
        # 选出有标签的index
        for data, label in tqdm(zip(category_data, category_label)):
            # 拆分训练集与测试集，需要打乱
            X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=opt.test_fraction, shuffle=True)
            # print(X_train.shape, y_train.shape)
            # print(X_test.shape, y_test.shape)

            np.concatenate((train_X, X_train), axis=0)
            np.concatenate((train_y, y_train), axis=0)
            np.concatenate((test_X, X_test), axis=0)
            np.concatenate((test_y, y_test), axis=0)

        # 训练数据集
        if train:
            # 最后需要的数据X与对应的标签y
            self.X = train_X
            self.y = train_y
            # print(self.X.shape)

        else:  # 测试数据集
            self.X = test_X
            self.y = test_y

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
        return len(self.X)

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