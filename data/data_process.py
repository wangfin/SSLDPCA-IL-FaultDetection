#!/usr/bin/env python
# @Time    : 2021/3/2 15:44
# @Author  : wb
# @File    : data_process.py

'''
数据处理页面，将数据处理成需要的格式
'''

import h5py
import pandas as pd
import numpy as np
import os
import scipy.io as scio
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import image
from pyts.image import GramianAngularField
from tqdm import tqdm
from tslearn.piecewise import PiecewiseAggregateApproximation

from config import opt

class DataProcess(object):
    '''
        处理CWRU，凯斯西储大学轴承数据
        CWRU原始数据分为驱动端与风扇端（DE，FE）
        正常 4个
        12K采样频率下的驱动端轴承故障数据 52个 没有第四种载荷的情况
        48K采样频率下的驱动端轴承故障数据*（删除不用）
        12K采样频率下的风扇端轴承故障数据 45个
        每个采样频率下面三种故障直径，每种故障直径下面四种电机载荷，每种载荷有三种故障
        内圈故障，外圈故障（三个位置），滚动体故障
        总共101个数据文件
    '''

    def CWRU_data_1d(self, type='DE'):
        '''
        直接处理1d的时序数据
        :type: DE或者FE，驱动端还是风扇端
        :return: 保存为h5文件
        '''

        # 维度
        dim = opt.CWRU_dim
        # CWRU原始数据
        CWRU_data_path = opt.CWRU_data_root
        read_file_directory = opt.read_file_directory
        # 一维数据保存路径
        save_path = opt.CWRU_data_1d_root

        # 读取文件列表
        frame_name = os.path.join(CWRU_data_path, read_file_directory)
        frame = pd.read_table(frame_name)

        # 数据
        signals = []
        # 标签
        labels = []
        # 数据块数量
        data_num = []

        for idx in range(len(frame)):
            mat_name = os.path.join(CWRU_data_path, frame['file_name'][idx])
            raw_data = scio.loadmat(mat_name)
            # raw_data.items() X097_DE_time 所以选取5:7为DE的
            for key, value in raw_data.items():
                if key[5:7] == type:
                    # 以dim的长度划分，有多少个数据块
                    sample_num = value.shape[0] // dim
                    # print('sample_num', sample_num)

                    # 数据取整
                    signal = value[0:dim * sample_num].reshape(1, -1)
                    # print('signals', signals.shape)
                    # 把数据分割成sample_num个数据块，(609,400,1)
                    signal_split = np.array(np.split(signal, sample_num, axis=1))

                    # 保存行向量
                    signals.append(signal_split)
                    # (123,)一维的label
                    labels.append(idx * np.ones(sample_num))
                    # 保存每个类别数据块的数量
                    data_num.append(sample_num)

        # squeeze删除维度为1的维度，(1,123)->(123,)
        # axis=0为纵向的拼接，axis=1为纵向的拼接
        # (13477200,)
        signals_np = np.concatenate(signals).squeeze()
        # (33693,)
        labels_np = np.concatenate(np.array(labels)).astype('uint8')
        data_num_np = np.array(data_num).astype('uint8')
        print(signals_np.shape, labels_np.shape, data_num_np.shape)

        # 保存为h5的文件
        file_name = os.path.join(save_path, 'CWRU_mini_' + type + str(len(frame)) + '.h5')
        f = h5py.File(file_name, 'w')
        # 数据
        f.create_dataset('data', data=signals_np)
        # 标签
        f.create_dataset('label', data=labels_np)
        # 每个类别的数据块数量
        f.create_dataset('data_num', data=data_num_np)
        f.close()

    def CWRU_data_2d_gaf(self, type='DE'):
        '''
        把CWRU数据集做成2d图像，使用Gramian Angular Field (GAF)，保存为png图片
        因为GAF将n的时序信号转换为n*n，这样导致数据量过大，采用分段聚合近似（PAA）转换压缩时序数据的长度
        97：243938
        :type: DE还是FE
        :return: 保存为2d图像
        '''

        # CWRU原始数据
        CWRU_data_path = opt.CWRU_data_root
        # 维度
        dim = opt.CWRU_dim

        # 读取文件列表
        frame_name = os.path.join(CWRU_data_path, 'annotations.txt')
        frame = pd.read_table(frame_name)

        # 保存路径
        save_path = os.path.join(opt.CWRU_data_2d_root, type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # gasf文件目录
        gasf_path = os.path.join(save_path, 'gasf')
        if not os.path.exists(gasf_path):
            os.makedirs(gasf_path)
        # gadf文件目录
        gadf_path = os.path.join(save_path, 'gadf')
        if not os.path.exists(gadf_path):
            os.makedirs(gadf_path)

        for idx in tqdm(range(len(frame))):
            # mat文件名
            mat_name = os.path.join(CWRU_data_path, frame['file_name'][idx])
            # 读取mat文件中的原始数据
            raw_data = scio.loadmat(mat_name)
            # raw_data.items() X097_DE_time 所以选取5:7为DE的
            for key, value in raw_data.items():
                if key[5:7] == type:
                    # dim个数据点一个划分，计算数据块的数量
                    sample_num = value.shape[0] // dim

                    # 数据取整，把列向量转换成行向量
                    signal = value[0:dim * sample_num].reshape(1, -1)
                    # PAA 分段聚合近似（PAA）转换
                    # paa = PiecewiseAggregateApproximation(n_segments=100)
                    # paa_signal = paa.fit_transform(signal)

                    # 按sample_num切分，每个dim大小
                    signals = np.split(signal, sample_num, axis=1)

                    for i in tqdm(range(len(signals))):
                        # 将每个dim的数据转换为2d图像
                        gasf = GramianAngularField(image_size=dim, method='summation')
                        signals_gasf = gasf.fit_transform(signals[i])
                        gadf = GramianAngularField(image_size=dim, method='difference')
                        signals_gadf = gadf.fit_transform(signals[i])

                        # 保存图像
                        filename_gasf = os.path.join(gasf_path, str(idx) + '.%d.png' % i)
                        image.imsave(filename_gasf, signals_gasf[0])
                        filename_gadf = os.path.join(gadf_path, str(idx) + '.%d.png' % i)
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

    # 此函数未完成
    def CWRU_data_2d_transform(self, type='DE'):
        '''
        使用数据拼接的方式，将一个长的时序数据拆分成小段，将小段按按行拼接
        如果直接进行拼接的话样本数量比较少，采用时间窗移动切割，也就是很多数据会重复
        这样可以提高图片的数量
        未完成
        :param type:DE or FE
        :return:
        '''
        # CWRU原始数据
        CWRU_data_path = opt.CWRU_data_root
        # 维度
        dim = opt.CWRU_dim

        # 读取文件列表
        frame_name = os.path.join(CWRU_data_path, 'annotations.txt')
        frame = pd.read_table(frame_name)

        # 保存路径
        save_path = os.path.join(opt.CWRU_data_2d_root, type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 转换生成的图像文件目录
        transform_path = os.path.join(save_path, 'transform')
        if not os.path.exists(transform_path):
            os.makedirs(transform_path)

        for idx in tqdm(range(len(frame))):
            # mat文件名
            mat_name = os.path.join(CWRU_data_path, frame['file_name'][idx])
            # 读取mat文件中的原始数据
            raw_data = scio.loadmat(mat_name)
            # raw_data.items() X097_DE_time 所以选取5:7为DE的
            for key, value in raw_data.items():
                if key[5:7] == type:
                    # dim个数据点一个划分，计算数据块的数量
                    sample_num = value.shape[0] // dim

                    # 数据取整，并转换为行向量
                    signal = value[0:dim * sample_num].reshape(1, -1)
                    # 归一化到[-1,1]，生成灰度图
                    signal = self.normalization(signal)

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
        '''
        归一化
        :param data:
        :return:
        '''
        _range = np.max(abs(data))
        return data / _range

    def png2h5(self):
        '''
        将保存好的png图片保存到h5文件中，需要大量内存
        :return: h5文件
        '''
        # 根目录
        img_root = opt.CWRU_data_2d_DE
        # 全部的图片的ID
        imgs_path = [os.path.join(img_root, img) for img in os.listdir(img_root)]
        # 图片数据
        imgs = []
        # 标签值
        labels = []
        for path in tqdm(imgs_path):
            img = Image.open(path)
            # img是Image内部的类文件，还需转换
            img_PIL = np.asarray(img, dtype='uint8')
            labels.append(path.split('/')[-1].split('\\')[-1].split('.')[0])
            imgs.append(img_PIL)
            # 关闭文件，防止多线程读取文件太多
            img.close()

        imgs = np.asarray(imgs).astype('uint8')
        labels = np.asarray(labels).astype('uint8')
        # 创建h5文件
        file = h5py.File(opt.CWRU_data_2d_h5, "w")
        # 在文件中创建数据集
        file.create_dataset("image", np.shape(imgs), dtype='uint8', data=imgs)
        # 标签
        file.create_dataset("label", np.shape(labels), dtype='uint8', data=labels)
        file.close()

    '''
        考虑下也导入田纳西-伊斯曼Tennessee Eastman(TE过程)的数据
        TE过程有21个故障，也就是21个dat文件（21个训练，21个测试）
        有53个变量，41个+12个
    '''
    # 此函数未完成
    def TE_data_1d(self):
        '''
        1D的TE过程数据处理
        :return:
        '''



if __name__ == '__main__':
    data = DataProcess()
    data.CWRU_data_1d(type='DE')

    # DE(33693, 400) (33693,)
    # FE(33693, 400) (33693,)





