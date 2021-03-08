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
from config import opt

class DataProcess(object):

    def CWRU_data(self):
        '''
            处理CWRU，凯斯西储大学轴承数据
            将CWRU原始数据分为驱动端与风扇端（DE，FE）
            12K采样频率下的驱动端轴承故障数据
            48K采样频率下的驱动端轴承故障数据*
            12K采样频率下的风扇端轴承故障数据
            每个采样频率下面三种故障直径，每种故障直径下面四种电机载荷，每种载荷有三种故障
            内圈故障，外圈故障（三个位置），滚动体故障
            总共101个数据文件
        :return: 保存为h5文件
        '''

        # 数据处理页面
        # 读取文件列表
        frame_name = os.path.join(opt.CWRU_data, 'annotations.txt')
        frame = pd.read_table(frame_name)
        # 维度
        dim = opt.CWRU_dim
        # 训练样本80%
        train_fraction = opt.train_fraction

        signals_tr = []
        labels_tr = []
        signals_tt = []
        labels_tt = []
        count = 0
        for idx in range(len(frame)):
            mat_name = os.path.join(opt.CWRU_data, frame['file_name'][idx])
            raw_data = scio.loadmat(mat_name)
            # raw_data.items() X097_DE_time 所以选取5:7为DE的
            for key, value in raw_data.items():
                if key[5:7] == 'FE':
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
                    signals_tt.append(signals[train_num:sample_num, :])
                    labels_tr.append(idx * np.ones(train_num))
                    labels_tt.append(idx * np.ones(test_num))

        signals_tr_np = np.concatenate(signals_tr).squeeze()  # 纵向的拼接，删除维度为1的维度
        labels_tr_np = np.concatenate(np.array(labels_tr)).astype('uint8')
        signals_tt_np = np.concatenate(signals_tt).squeeze()
        labels_tt_np = np.concatenate(np.array(labels_tt)).astype('uint8')
        print(signals_tr_np.shape, labels_tr_np.shape, signals_tt_np.shape, labels_tt_np.shape)

        # 保存为h5的文件
        f = h5py.File('CWRU_FE.h5', 'w')
        f.create_dataset('X_train', data=signals_tr_np)
        f.create_dataset('y_train', data=labels_tr_np)
        f.create_dataset('X_test', data=signals_tt_np)
        f.create_dataset('y_test', data=labels_tt_np)
        f.close()


if __name__ == '__main__':
    data = DataProcess()
    data.CWRU_data()



