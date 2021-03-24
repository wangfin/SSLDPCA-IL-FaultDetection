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
from scipy.spatial.distance import pdist
from collections import Counter

from config import opt
from SSLDPCA.plot import Plot

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
       7.调整数据格式，输出所有的数据（大量伪标签，少量真实标签）
       density distance
    '''

    def __init__(self):
        '''
        读取处理好的1d数据文件
        ../data/CWRU_data_1d/CWRU_DE.h5
        '''
        # h5文件路径
        file_path = '../data/CWRU_data_1d/CWRU_mini_DE.h5'
        # 读取数据
        f = h5py.File(file_path, 'r')
        # 数据，取值，可以用f['data'].value，不过包自己推荐使用f['data'][()]这种方式
        self.data = f['data'][()]
        # 标签
        self.label = f['label'][()]
        # 程序生成的伪标签
        self.pseudo_label = []
        # 每个类别的数据块数量
        self.data_num = f['data_num'][()]

        # 有标签数据的占比
        self.label_fraction = opt.label_fraction
        # 故障的类别
        self.category = opt.CWRU_category
        # 邻居数量，因为计算的K邻居的第一个是自己，所以需要+1
        self.neighbor_num = opt.K + 1
        # 数据的维度
        self.dim = opt.CWRU_dim
        # K邻居模型保存路径
        self.K_neighbor = './K-neighbor_mini.h5'
        # 间隔超参数
        self.lambda_delta = opt.lambda_delta

    def label_data(self):
        '''
        处理有标签的数据，选择其中的一部分作为标签数据输入算法，其他数据的标签全部清除
        :return: label_datas，每个类别的有标签的数据集合 [[],[],...,[]]
        '''
        # 选取一定比例的有标签数据（这个是仿真中可以调整的参数）
        # 为了实验起见，选取平衡的数据，即每个类分配相同比例的有标签数据集
        # (33693,400)(33693)
        # 各个类别的数据
        category_data = []
        # 有标签的数据
        label_datas = []

        # 把每个类别的数据切分出来
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
            knn_model.to_hdf5(self.K_neighbor)

        return knn_model

    def neighbors(self):
        '''
        对输入的data进行计算邻居与距离，data形式为[[],[],...,[]]
        :param data:
        :return:
        '''
        starttime = datetime.datetime.now()
        # 在导入h5模型之前还是需要构建一下模型
        knn_model = neighbors.KNeighborsTimeSeries(n_neighbors=self.neighbor_num,
                                                   metric='euclidean',
                                                   n_jobs=-1)
        # 如果存在保存好的模型，h5
        if os.path.exists(self.K_neighbor):
            knn_model = knn_model.from_hdf5(self.K_neighbor)
            # 还是需要数据拟合一下
            knn_model.fit(self.data)
        else:
            # 没有保存模型那就去模型函数那边训练模型
            knn_model = self.neighbors_model()
        # 需要计算邻居的数据
        test_time_series = to_time_series_dataset(self.data)
        # K邻居的距离和邻居的id
        distance, neighbors_index = knn_model.kneighbors(test_time_series, return_distance=True)

        endtime = datetime.datetime.now()
        print('计算邻居用时', (endtime - starttime).seconds)
        return distance, neighbors_index

    def build_density(self, distance, neighbors_index):
        '''
        计算每个数据点的密度
        使用SNN的方式，共享邻居数据点
        两个邻居数据点才有相似度，相似度公式为S/d(i)+d(j)，每个数据点的密度是K邻居的相似度之和
        dis和nei的第一个点都是自己本身，需要去掉
        :param: distance,[[],[],[]]数据点的距离
        :param: neighbors_index,[[],[],[]]数据点的邻居列表
        :return:密度列表，[, ,...,]
        '''
        starttime = datetime.datetime.now()
        # 每个数据点的密度
        density = []
        for index in range(len(distance)):
            # 该数据点的平均邻居距离，去掉第一个点，第一个是本身数据点
            node_distance_avg = np.mean(distance[index][1:])

            # 数据点的密度
            node_density = 0

            # 从一个数据点的邻居内开始计算，neighbor是邻居的ID
            for neighbor in neighbors_index[index]:
                # 求该数据点的邻居与邻居的邻居有多少是重复邻居
                snn = list(set(neighbors_index[index][1:]).intersection(set(neighbors_index[neighbor][1:])))
                # 共享邻居的数量
                snn_num = len(snn)
                # 邻居数据点的平均距离
                neighbors_distance_avg = np.mean(distance[neighbor][1:])

                # 两个数据点的相似度
                sim = snn_num / (node_distance_avg + neighbors_distance_avg)
                # 数据点的密度是每个邻居的相似度的和
                node_density += sim

            # 所有数据点的密度
            density.append(node_density)

        endtime = datetime.datetime.now()
        print('计算密度用时', (endtime - starttime).seconds)

        return density

    def build_interval(self, density, distance, label_datas):
        '''
        计算每个数据点的间隔，基于密度和距离
        :param density: 密度
        :param distance: 邻居距离
        :param label_datas: 注入的有标签的数据
        :return:interval []
        '''
        # 1.首先需要寻找到比数据点密度更高的数据点
        # 2.然后计算dij，i的平均邻居距离，j的平均邻居距离，i到某一个类别有标签样本点的最小平均距离
        # 3.密度最大值的数据点需要成为最高的间隔值

        starttime = datetime.datetime.now()

        # 数据点的间隔值
        interval = []
        # 因为排序过，所以得换一种dict
        interval_dict = {}

        # 排序，获得排序的ID[]
        sort_density_idx = np.argsort(density)
        # 数据点node的index
        for node_i in range(len(sort_density_idx)):
            # 计算数据点node到有标签样本的平均距离，然后取最小的平均类别距离
            label_dis = []
            # 每个类别的有标签样本
            for category in label_datas:
                category_dis = []
                for i in range(len(category)):
                    # 两点间的距离
                    dis = self.euclidean_distance(self.data[sort_density_idx[node_i]], category[i])
                    category_dis.append(dis)
                # 每个类别的平均距离
                label_dis.append(np.mean(category_dis))
            # 到最近的类别的有标签样本数据的距离
            min_label_dis = np.min(label_dis)
            # 最近的类别
            # min_label = np.argmin(label_dis)

            # node_i的平均邻居距离
            node_i_distance_avg = np.mean(distance[sort_density_idx[node_i]])

            # 数据点的全部间隔
            node_intervals = []
            # 密度比node更大的数据点
            for node_j in range(node_i + 1, len(sort_density_idx)):
                # i，j的距离
                dij = self.euclidean_distance(self.data[sort_density_idx[node_i]], self.data[sort_density_idx[node_j]])
                # 数据点j的平均邻居距离
                node_j_distance_avg = np.mean(distance[sort_density_idx[node_j]])
                delta = (dij * (node_i_distance_avg + node_j_distance_avg))/min_label_dis
                node_intervals.append(delta)

            # 添加到interval
            # 判断node_intervals是否为空
            if node_intervals:
                # 不为空就是正常的间隔值
                # 因为排序过，所以不能是直接append，而是要找到位置入座
                interval_dict[sort_density_idx[node_i]] = np.min(node_intervals)
            else:
                # 如果为空，应该是密度最大值，先设置为-1，后面会为他设置为间隔最高值
                interval_dict[sort_density_idx[node_i]] = -1

        # 密度最高的数据点的间隔必须为间隔最大值
        # 这里用的是dict，所以需要先取出values，然后转换成list，才能使用np.max
        interval_dict[sort_density_idx[-1]] = np.max(list(interval_dict.values()))

        # 然后将dict按key排序，也就是回到从1-n的原序状态
        # 然后就可以把dict中的value输入到interval
        for key, value in sorted(interval_dict.items()):
            interval.append(value)

        endtime = datetime.datetime.now()
        print('计算间隔用时', (endtime - starttime).seconds)
        return interval

    def euclidean_distance(self, data1, data2):
        '''
        计算两个数据点之间的欧几里得距离
        :param n1: 数据1
        :param n2: 数据2
        :return: 距离
        '''

        X = np.vstack([data1, data2])
        distance = pdist(X, 'euclidean')[0]
        return distance

    def score(self, density, interval):
        '''
        根据数据点的密度与间隔，计算分数
        :param density: 数据点密度列表 []
        :param interval: 数据点间隔列表 []
        :return: scores [] 数据点的分数
        '''

        starttime = datetime.datetime.now()

        scores = []
        max_rho = np.max(density)
        max_delta = np.max(interval)

        for rho, delta in zip(density, interval):
            # 每个数据点的得分计算
            score = (rho / max_rho) * (delta / max_delta)
            scores.append(score)

        endtime = datetime.datetime.now()
        print('计算得分用时', (endtime - starttime).seconds)

        return scores

    def select_head(self, scores):
        '''
        根据每个数据点的分数，选择簇头
        :param scores: 数据点分数
        :return: 簇节点的ID heads []
        '''
        starttime = datetime.datetime.now()

        # 降序排序，需要选取分数最大的作为簇头
        score_index = np.argsort(-np.array(scores))
        # 有多少个故障类别，就有多少个簇头
        heads = score_index[:self.category].tolist()

        endtime = datetime.datetime.now()
        print('计算簇头用时', (endtime - starttime).seconds)

        return heads

    def divide_area(self, density, interval):
        '''
        为所有的无标签样本点分配标签
        :return:areas [[core_region], [border_region], [new_category_area]]
        '''
        # 1.在rho和delta的决策图中划分区域
        # 2.把所有的无标签点分配到这些区域
        # 3.输出每个区域内的数据点ID

        starttime = datetime.datetime.now()

        # 密度的分割线，平均密度
        rho_split_line = np.mean(density)
        # 间隔的分割线，lambda*间隔的方差
        delta_split_line = self.lambda_delta * np.var(interval)

        # 根据分割线划分区域
        # 核心区域
        core_region = []
        # 边缘区域
        border_region = []
        # 新类别区域
        new_category_area = []
        # 数据ID
        index = 0
        for rho, delta in zip(density, interval):

            if rho >= rho_split_line:
                core_region.append(index)
            elif rho < rho_split_line and delta < delta_split_line:
                border_region.append(index)
            elif rho < rho_split_line and delta >= delta_split_line:
                new_category_area.append(index)
            else:
                print('没这种数据')

            index = index + 1

        # 最后输出的三个区域的值
        areas = [core_region, border_region, new_category_area]

        endtime = datetime.datetime.now()
        print('划分区域用时', (endtime - starttime).seconds)

        return areas

    def assign_labels(self, heads, areas, label_datas):
        '''
        在划分完区域之后，开始对每个区域内的数据进行分配伪标签
        :param heads: 簇头的ID []
        :param areas: 每个区域内的数据点ID [[], [], []]
        :param label_datas: 有标签的数据ID [[],[],...,[]]
        :return:pseudo_labels 输出的伪标签
        '''

        starttime = datetime.datetime.now()

        # 核心区域
        core_region = areas[0]
        # 边缘区域
        border_region = areas[1]
        # 新类别区域
        new_category_region = areas[2]

        # 簇中心点的标签
        heads_labels = []
        # 簇中心点分配标签的过程
        for head_node in heads:
            # 计算数据点node到有标签样本的平均距离，然后取最小的平均类别距离
            label_dis = []
            # 每个类别的有标签样本
            for category in label_datas:
                category_dis = []
                for i in range(len(category)):
                    # 两点间的距离
                    dis = self.euclidean_distance(self.data[head_node], category[i])
                    category_dis.append(dis)
                # 每个类别的平均距离
                label_dis.append(np.mean(category_dis))
            # 最近的类别
            min_label = np.argmin(label_dis)
            heads_labels.append(min_label)

        # 核心区域的标签分配
        core_labels = []
        for core_node in core_region:
            head_dis = []
            for head_node in heads:
                # 核心区域中的点与簇中心点的距离
                dis = self.euclidean_distance(self.data[core_node], self.data[head_node])
                head_dis.append(dis)
            # 核心区域点的标签值与最近的簇中心点保持一致
            core_label = heads_labels[int(np.argmin(head_dis))]
            core_labels.append(core_label)

        # 边缘区域的标签分配
        border_labels = []
        for border_node in border_region:
            # 计算距离
            border_node_dis = []
            for core_node in core_region:
                # 边缘区域中的点与核心区域内点的距离
                dis = self.euclidean_distance(self.data[border_node], self.data[core_node])
                border_node_dis.append(dis)
            # 保存K邻居的标签值
            K_labels = []
            # 找到距离边缘点最近的核心点
            for i in np.argsort(border_node_dis)[:opt.K]:
                K_labels.append(core_labels[i])

            # 这里是dict，Counter({3: 2, 10: 2, 1: 1, 0: 1})
            max_K_labels = Counter(K_labels)
            # 按value对dict排序，逆序排序
            max_K_labels = sorted(max_K_labels.items(), key=lambda item: item[1], reverse=True)
            # max_K_labels[0]为最大值，max_K_labels[0][0]为最大值的key
            border_labels.append(max_K_labels[0][0])

        # 新类别区域的标签分配
        # new_category_labels = []
        new_category_labels = self.new_category_label(new_category_region)

        # 最后需要把标签按顺序摆好，然后输出
        pseudo_labels = []
        # 把几个list合并一下
        data_index = heads + core_region + border_region + new_category_region
        data_labels = heads_labels + core_labels + border_labels + new_category_labels
        # 设置一个dict
        pseudo_labels_dict = {}
        for i in range(len(data_index)):
            # 给这个dict赋值
            pseudo_labels_dict[data_index[i]] = data_labels[i]

        # 然后将dict按key排序，也就是回到从1-n的原序状态
        # 然后就可以把dict中的value输入到pseudo_labels
        for key, value in sorted(pseudo_labels_dict.items()):
            pseudo_labels.append(value)

        endtime = datetime.datetime.now()
        print('分配标签用时', (endtime - starttime).seconds)

        return pseudo_labels


    def new_category_label(self, new_category_area):
        '''
        对新类别区域的数据样本点分配标签
        :param new_category_area: 新类别区域的数据
        :return: new_category_labels 新类别区域中样本点的标签
        '''

        # 这边全部定的是-1
        # 长度就是new_category_area的长度
        new_category_labels = [-1 for _ in range(len(new_category_area))]

        return new_category_labels


if __name__ == '__main__':
    ssldpca = SslDpca1D()
    # 选出有标签的数据，准备注入
    label_datas = ssldpca.label_data()
    # ssldpca.neighbors_model()
    distance, neighbors_index = ssldpca.neighbors()
    density = ssldpca.build_density(distance, neighbors_index)
    interval = ssldpca.build_interval(density, distance, label_datas)
    # print(density)
    # print(interval)
    #
    scores = ssldpca.score(density, interval)
    heads = ssldpca.select_head(scores)
    #
    areas = ssldpca.divide_area(density, interval)
    # print(areas)
    #
    pseudo_labels = ssldpca.assign_labels(heads, areas, label_datas)
    # print(pseudo_labels)

    # plot = Plot()
    # # plot.plot_data(ssldpca.data, ssldpca.label)
    # plot.plot_areas(ssldpca.data, areas)




