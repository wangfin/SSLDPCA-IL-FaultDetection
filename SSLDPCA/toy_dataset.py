#!/usr/bin/env python
# @Time    : 2021/4/14 9:51
# @Author  : wb
# @File    : toy_dataset.py


import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from itertools import cycle, islice
from scipy.spatial.distance import pdist

from sklearn.datasets import make_moons, make_blobs, make_circles

class ToyDataset():

    def moon_data(self):
        '''
        读取sklearn two_moons数据集
        :return: X n*2 y
        '''

        # noise 噪声，高斯噪声的标准偏差
        X, y = make_moons(n_samples=1000, noise=0.06)
        X = StandardScaler().fit_transform(X)
        return X, y

    def blobs_data(self):
        '''
        读取sklearn blobs数据集
        :return: X n*2 y
        '''
        # n_features 特征维度
        X, y = make_blobs(n_samples=1200, n_features=3, centers=6, cluster_std=0.6, center_box=(-7, 15))
        X = StandardScaler().fit_transform(X)
        return X, y

    def circles_data(self):
        '''
        读取sklearn circles数据集
        :return:
        '''
        # factor 内圆和外圆之间的比例因子，范围为（0，1）。
        X, y = make_circles(n_samples=2000, noise=0.07, factor=0.5)
        X = StandardScaler().fit_transform(X)
        return X, y

    def jain_data(self):
        '''
        读取Jain的数据
        :return:
        '''
        X = []
        y = []
        jain_path = '../data/Jain.txt'
        with open(jain_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            split = line.split()
            X.append([float(split[0]), float(split[1])])
            y.append(int(split[2]))

        return np.array(X), np.array(y)

    def pathbased_data(self):
        '''
        读取 Pathbased 数据集
        :return:
        '''
        X = []
        y = []
        jain_path = '../data/Pathbased.txt'
        with open(jain_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            split = line.split()
            X.append([float(split[0]), float(split[1])])
            y.append(int(split[2]))

        return np.array(X), np.array(y)

    def neighbors(self, data, n_neighbors):
        '''
        从输入数据中找到邻居
        :param: data 数据
        :return:
        '''
        neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=0.4, n_jobs=-1)
        neigh.fit(data)
        neigh_dist, neigh_ind = neigh.kneighbors()

        return neigh_dist, neigh_ind

    def build_density(self, neigh_dist, neigh_ind):
        '''
        计算每个数据点的密度
        使用SNN的方式，共享邻居数据点
        两个邻居数据点才有相似度，相似度公式为S/d(i)+d(j)，每个数据点的密度是K邻居的相似度之和
        :param: neigh_ind 邻居id
        :param: neigh_dist 邻居的距离
        :return:density 每个点的密度值
        '''
        density = []

        # 平均邻居距离
        neigh_dist_avg = []
        for dist in neigh_dist:
            neigh_dist_avg.append(np.mean(dist))

        # 共享邻居数量
        for neigh_index in enumerate(neigh_ind):
            # 每个节点的密度值，是相似度的和
            node_density = 0
            # 这个点的邻居列表
            for neighbor in neigh_index[1]:
                # 求该数据点的邻居与邻居的邻居有多少是重复邻居
                snn = list(set(neigh_index[1]).intersection(set(neigh_ind[neighbor])))
                # 共享邻居的数量
                snn_num = len(snn)
                # 求个平方
                snn_num = np.square(snn_num)

                # 两个数据点的相似度
                sim = snn_num / (neigh_dist_avg[neigh_index[0]] + neigh_dist_avg[neighbor])
                # 数据点的密度是每个邻居的相似度的和
                node_density += sim

            # 所有数据点的密度
            density.append(node_density)

        return density

    def build_interval(self, data, density, neigh_dist):
        '''
        计算每个数据点的间隔
        :param density: 密度
        :param neigh_dist: 邻居距离
        :param label_datas: 有标签的值
        :return:
        '''
        # 数据点的间隔值
        interval = []
        # 因为排序过，所以得换一种dict
        interval_dict = {}

        # 平均邻居距离
        neigh_dist_avg = []
        for dist in neigh_dist:
            neigh_dist_avg.append(np.mean(dist))

        # 排序，获得排序的ID[]
        sort_density_idx = np.argsort(density)

        # 数据点node的index
        for node_i in range(len(sort_density_idx)):
            # 数据点的全部间隔
            node_intervals = []
            # 密度比node更大的数据点
            for node_j in range(node_i + 1, len(sort_density_idx)):
                # i，j的距离
                dij = self.euclidean_distance(data[sort_density_idx[node_i]], data[sort_density_idx[node_j]])
                # dij*(node_i的平均邻居值+node_j的平均邻居值)
                delta = (dij * (neigh_dist_avg[sort_density_idx[node_i]] + neigh_dist_avg[sort_density_idx[node_j]]))
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

        return interval

    def score(self, density, interval):
        '''
        计算数据点的得分
        :param density: 密度
        :param interval: 间隔
        :return: scores 得分
        '''
        scores = []
        max_rho = np.max(density)
        max_delta = np.max(interval)

        for rho, delta in zip(density, interval):
            # 每个数据点的得分计算
            score = (rho / max_rho) * (delta / max_delta)
            scores.append(score)
        return scores

    def select_head(self, scores, class_num):
        '''
        根据每个数据点的分数，选择簇头
        :param scores: 数据点分数
        :param class_num: 类别数
        :return: 簇节点的ID heads []
        '''

        # 降序排序，需要选取分数最大的作为簇头
        score_index = np.argsort(-np.array(scores))
        # 有多少个故障类别，就有多少个簇头
        heads = score_index[:class_num].tolist()

        return heads

    def divide_area(self, density, interval, lambda_delta):
        '''
        划分区域
        :param density:
        :param interval:
        :param lambda_delta:
        :return: areas [[core_region], [border_region], [new_category_region]]
        '''
        # 1.在rho和delta的决策图中划分区域
        # 2.把所有的无标签点分配到这些区域
        # 3.输出每个区域内的数据点ID

        # 密度的分割线，平均密度
        rho_split_line = np.mean(density)
        # 间隔的分割线，lambda*间隔的方差
        delta_split_line = lambda_delta * np.var(interval)

        # 根据分割线划分区域
        # 核心区域
        core_region = []
        # 边缘区域
        border_region = []
        # 新类别区域
        new_category_region = []
        # 数据ID
        index = 0
        for rho, delta in zip(density, interval):

            if rho >= rho_split_line:
                core_region.append(index)
            elif rho < rho_split_line and delta < delta_split_line:
                border_region.append(index)
            elif rho < rho_split_line and delta >= delta_split_line:
                new_category_region.append(index)
            else:
                print('没这种数据')

            index = index + 1

        # 最后输出的三个区域的值
        areas = [core_region, border_region, new_category_region]

        return areas

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

    def plot_data(self, X, y):
        '''
        绘制二维数据，是二维
        :param X: 数据X
        :param y: 标签y
        :return:
        '''
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y) + 1))))

        plt.scatter(X[:, 0], X[:, 1], s=3, color=colors[y])
        plt.show()

    def plot_heads(self, X, y, heads):
        '''
        绘制带簇头的二维数据
        :param X: 数据X
        :param y: 标签y
        :param heads: 簇头
        :return:
        '''
        plt.figure(figsize=(8, 6))
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y) + 1))))

        plt.scatter(X[:, 0], X[:, 1], s=3, color=colors[y])
        for head in heads:
            plt.scatter(X[head, 0], X[head, 1], s=10, color='k', marker='*')

        plt.show()

    def plot_rho_delta(self, density, interval):
        '''
        绘制rho-delta 密度间隔决策图
        :param density: 密度
        :param interval: 间隔
        :return:
        '''
        plt.figure(figsize=(8, 6))
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y) + 1))))

        # 添加X/Y轴描述
        plt.xlabel('rho')
        plt.ylabel('delta')

        plt.scatter(density, interval, s=3, color=colors[y])
        for head in heads:
            plt.scatter(density[head], interval[head], s=15, color='k', marker='^')

        plt.show()

if __name__ == '__main__':
    toy = ToyDataset()
    X, y = toy.pathbased_data()

    n_neighbors = int(len(X) * 0.2)
    neigh_dist, neigh_ind = toy.neighbors(X, n_neighbors)
    # print(neigh_dist)
    density = toy.build_density(neigh_dist, neigh_ind)
    # print(density)
    interval = toy.build_interval(X, density, neigh_dist)
    # print(interval)

    scores = toy.score(density, interval)
    heads = toy.select_head(scores, class_num=3)

    toy.plot_heads(X, y, heads)
    toy.plot_rho_delta(density, interval)


