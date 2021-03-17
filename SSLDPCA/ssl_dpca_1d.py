#!/usr/bin/env python
# @Time    : 2021/3/16 21:28
# @Author  : wb
# @File    : ssl_dpca_1d.py
import datetime
import os

import h5py
from tqdm import tqdm
import random
import numpy as np
from tslearn import neighbors
from tslearn.utils import to_time_series_dataset

from config import opt

'''
半监督（SSL）的密度峰值聚类（DPCA），此文件用于1d数据
'''

class SslDpca1D(object):
    '''
       半监督的DPCA，在原始的DPCA的基础上加入半监督（小部分有标签数据）
       步骤：
       1.注入部分的有标签数据
       2.计算密度与间隔
       3.计算数据点的分数
       4.选取分数高的作为簇中心
       5.根据规则在 密度-间隔决策图 上划分区域
       6.对每个区域内的数据点分配标签
           6.1 簇中心点的标签由簇中心点到带有真实标签样本数据的距离决定
           6.2 核心区域中的主干点分配给距离最近的簇中心点，并且主干点的标签与所属簇中心点的标签保持一致
           6.3 边缘区域内的边缘点选择与它距离最近K个主干点的标签值，K是人为设定值
           6.4 新类别的样本数据点需要人为标注一部分的标签，然后使用Kmeans聚类传递标签
       7.输出所有的数据（大量伪标签，少量真实标签）
       density distance
    '''

    def __init__(self):
        '''
        读取处理好的1d数据文件
        ../data/CWRU_data_1d/CWRU_DE.h5
        '''
        # h5文件路径
        file_path = '../data/CWRU_data_1d/CWRU_DE.h5'
        # 读取数据
        f = h5py.File(file_path, 'r')
        # 数据，取值，可以用f['data'].value，不过包自己推荐使用f['data'][()]这种方式
        self.data = f['data'][()]
        # 标签
        self.label = f['label'][()]
        # 每个类别的数据块数量
        self.data_num = f['data_num'][()]

        # 有标签数据的占比
        self.label_fraction = opt.label_fraction
        # 故障的类别
        self.category = opt.CWRU_category
        # 邻居数量
        self.neighbor_num = opt.K
        # 数据的维度
        self.dim = opt.CWRU_dim
        # K邻居模型保存路径
        self.K_neighbor = './K-neighbor.h5'

    def label_data(self):
        '''
        处理有标签的数据，选择其中的一部分作为标签数据输入算法，其他数据的标签全部清除
        :return: input_label_index，每个类别的有标签的数据的ID集合
        '''
        # 选取一定比例的有标签数据（这个是仿真中可以调整的参数）
        # 为了实验起见，选取平衡的数据，即每个类分配相同比例的有标签数据集
        # (33693,400)(33693)
        # 各个类别的数据
        category_data = []
        # 有标签的数据
        label_datas = []

        point = 0
        for i in range(len(self.data_num)):
            data = self.data[point:point + self.data_num[i]]
            point = point + self.data_num[i]
            category_data.append(data)

        # 选出有标签的index
        for category in tqdm(category_data):
            # 有标签的数量
            label_num = int(len(category) * self.label_fraction)
            # 对category的格式为(609,400)
            # 随机从数据中取出数量的值，需要先转换为list
            label_data = random.sample(category.tolist(), label_num)
            # 再把list转换为ndarray
            label_datas.append(np.array(label_data))

        return label_datas

    def neighbors_model(self):
        '''
        计算数据的K邻居
        :return:
        '''
        # 把数据凑成需要的维度
        # tslearn的三个维度，分别对应于时间序列的数量、每个时间序列的测量数量和维度的数量
        # 使用tslearn计算K邻居
        train_time_series = to_time_series_dataset(self.data)
        knn_model = neighbors.KNeighborsTimeSeries(n_neighbors=self.neighbor_num,
                                                   metric='euclidean',
                                                   n_jobs=-1)
        knn_model.fit(train_time_series)
        if not os.path.exists(self.K_neighbor):
            knn_model.to_hdf5('./K-neighbor.h5')

        return knn_model

    def neighbors(self):
        '''
        对输入的data进行计算邻居与距离，data形式为[[],[],...,[]]
        :param data:
        :return:
        '''
        knn_model = neighbors.KNeighborsTimeSeries(n_neighbors=self.neighbor_num,
                                                   metric='euclidean',
                                                   n_jobs=-1)
        if os.path.exists(self.K_neighbor):
            knn_model = knn_model.from_hdf5(self.K_neighbor)
            knn_model.fit(self.data)
        else:
            knn_model = self.neighbors_model()
        starttime = datetime.datetime.now()
        test_time_series = to_time_series_dataset(self.data)
        # K邻居的距离和邻居的id
        distance, neighbors_index = knn_model.kneighbors(test_time_series, return_distance=True)
        endtime = datetime.datetime.now()
        print((endtime - starttime).seconds)
        return distance, neighbors_index

if __name__ == '__main__':
    ssldpca = SslDpca1D()
    # ssldpca.neighbors_model()
    distance, neighbors_index = ssldpca.neighbors()
    print(neighbors_index)