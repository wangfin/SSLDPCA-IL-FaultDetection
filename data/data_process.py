#!/usr/bin/env python
# @Time    : 2021/3/2 15:44
# @Author  : wb
# @File    : data_process.py

'''
数据处理页面
'''

import h5py
import pandas as pd
import numpy as np
import os
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib import image
from pyts.image import GramianAngularField
from tqdm import tqdm

from config import opt

class DataProcess(object):

    def CWRU_data(self, type='DE'):
        '''
            处理CWRU，凯斯西储大学轴承数据
            将CWRU原始数据分为驱动端与风扇端（DE，FE）
            12K采样频率下的驱动端轴承故障数据
            48K采样频率下的驱动端轴承故障数据*
            12K采样频率下的风扇端轴承故障数据
            每个采样频率下面三种故障直径，每种故障直径下面四种电机载荷，每种载荷有三种故障
            内圈故障，外圈故障（三个位置），滚动体故障
            总共101个数据文件
        :type: DE或者FE，驱动端还是风扇端
        :return: 保存为h5文件
        '''

        # 数据处理页面

        # 维度
        dim = opt.CWRU_dim
        # CWRU原始数据
        CWRU_data_path = opt.CWRU_data
        # h5保存路径
        save_path = opt.CWRU_data_1d
        # 训练样本80%
        train_fraction = opt.train_fraction

        # 读取文件列表
        frame_name = os.path.join(CWRU_data_path, 'annotations.txt')
        frame = pd.read_table(frame_name)

        signals_tr = []
        labels_tr = []
        signals_te = []
        labels_te = []
        for idx in range(len(frame)):
            mat_name = os.path.join(CWRU_data_path, frame['file_name'][idx])
            raw_data = scio.loadmat(mat_name)
            # raw_data.items() X097_DE_time 所以选取5:7为DE的
            for key, value in raw_data.items():
                if key[5:7] == type:
                    signal = value
                    # print(signal.shape)
                    sample_num = signal.shape[0] // dim
                    # print('sample_num', sample_num)

                    # 这边是将一维向量转换成[signal.shape[0]//dim,dim]的数组
                    train_num = int(sample_num * train_fraction)
                    test_num = sample_num - train_num

                    signal = signal[0:dim * sample_num]
                    # 按sample_num切分
                    signals = np.array(np.split(signal, sample_num))
                    # print('signals', signals.shape)

                    signals_tr.append(signals[0:train_num, :])
                    signals_te.append(signals[train_num:sample_num, :])
                    labels_tr.append(idx * np.ones(train_num))
                    labels_te.append(idx * np.ones(test_num))

        signals_tr_np = np.concatenate(signals_tr).squeeze()  # 纵向的拼接，删除维度为1的维度
        labels_tr_np = np.concatenate(np.array(labels_tr)).astype('uint8')
        signals_te_np = np.concatenate(signals_te).squeeze()
        labels_te_np = np.concatenate(np.array(labels_te)).astype('uint8')
        print(signals_tr_np.shape, labels_tr_np.shape, signals_te_np.shape, labels_te_np.shape)

        # 保存为h5的文件
        file_name = os.path.join(save_path, 'CWRU_' + type + '.h5')
        f = h5py.File(file_name, 'w')
        f.create_dataset('X_train', data=signals_tr_np)
        f.create_dataset('y_train', data=labels_tr_np)
        f.create_dataset('X_test', data=signals_te_np)
        f.create_dataset('y_test', data=labels_te_np)
        f.close()

    def CWRU_data_2d(self, type='DE'):
        '''
        把CWRU数据集做成2d图像，使用Gramian Angular Field (GAF)
        97：243938
        :type: DE还是FE
        :return: 保存为2d图像
        '''

        # 数据处理页面
        # 读取文件列表
        frame_name = os.path.join(opt.CWRU_data, 'annotations.txt')
        frame = pd.read_table(frame_name)
        # 维度
        dim = opt.CWRU_dim
        # 保存路径
        save_path = os.path.join(opt.CWRU_data_2d, type)

        for idx in tqdm(range(len(frame))):
            # mat文件名
            mat_name = os.path.join(opt.CWRU_data, frame['file_name'][idx])
            # 读取mat文件中的原始数据
            raw_data = scio.loadmat(mat_name)
            # raw_data.items() X097_DE_time 所以选取5:7为DE的
            for key, value in raw_data.items():
                if key[5:7] == type:
                    # dim个数据点一个划分，计算数据块的数量
                    sample_num = value.shape[0] // dim

                    # 数据取整
                    signal = value[0:dim * sample_num]
                    # 按sample_num切分，每个dim大小
                    signals = np.array(np.split(signal, sample_num))

                    for i in tqdm(range(sample_num)):
                        # 将每个dim的数据转换为2d图像
                        gasf = GramianAngularField(image_size=dim, method='summation')
                        signals_gasf = gasf.fit_transform(signals[i].reshape(1, -1))
                        gadf = GramianAngularField(image_size=dim, method='difference')
                        signals_gadf = gadf.fit_transform(signals[i].reshape(1, -1))

                        # 保存图像
                        filename_gasf = os.path.join(save_path, 'gasf', str(idx) + '.%d.png' % i)
                        image.imsave(filename_gasf, signals_gasf[0])
                        filename_gadf = os.path.join(save_path, 'gadf', str(idx) + '.%d.png' % i)
                        image.imsave(filename_gadf, signals_gadf[0])

                    # 展示图片
                    # images = [signals_gasf[0], signals_gadf[0]]
                    # titles = ['Summation', 'Difference']
                    #
                    # fig, axs = plt.subplots(1, 2, constrained_layout=True)
                    # for image, title, ax in zip(images, titles, axs):
                    #     ax.imshow(image)
                    #     ax.set_title(title)
                    # fig.suptitle('GramianAngularField', y=0.94, fontsize=16)
                    # plt.margins(0, 0)
                    # plt.savefig("GramianAngularField.pdf", pad_inches=0)
                    # plt.show()

    def CWRU_data_2d_transform(self, type='DE'):
        '''
        使用数据拼接的方式，将一个长的时序数据拆分成小段，将小段按按行拼接
        :param type:
        :return:
        '''
        # 数据处理页面
        # 读取文件列表
        frame_name = os.path.join(opt.CWRU_data, 'annotations.txt')
        frame = pd.read_table(frame_name)
        # 维度
        dim = opt.CWRU_dim
        # 保存路径
        save_path = os.path.join(opt.CWRU_data_2d, type)

        for idx in tqdm(range(1)):
            # mat文件名
            mat_name = os.path.join(opt.CWRU_data, frame['file_name'][idx])
            # 读取mat文件中的原始数据
            raw_data = scio.loadmat(mat_name)
            # raw_data.items() X097_DE_time 所以选取5:7为DE的
            for key, value in raw_data.items():
                if key[5:7] == type:
                    # dim个数据点一个划分，计算数据块的数量
                    sample_num = value.shape[0] // dim

                    # 数据取整
                    signal = value[0:dim * sample_num]
                    # 归一化到[-1,1]，生成灰度图
                    signal = self.normalization(signal)
                    # 转换成行向量
                    signal = np.array(signal).reshape(1, -1)
                    # 按sample_num切分，每一个块dim大小
                    signals = np.split(signal, sample_num, axis=1)

                    # 生成正方形的图片，正方形面积小，能生成多张图片
                    pic_num = sample_num // dim
                    pic_data = []
                    for i in range(pic_num-1):
                        pic_data.append(signals[i * dim:(i + 1) * dim])

                        # pic = np.concatenate(pic_data).squeeze()

                        # 展示图片
                        plt.imshow(pic_data)
                        plt.show()

    def normalization(self, data):
        _range = np.max(abs(data))
        return data / _range




if __name__ == '__main__':
    data = DataProcess()
    data.CWRU_data_2d()




