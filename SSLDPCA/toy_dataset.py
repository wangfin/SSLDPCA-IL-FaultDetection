#!/usr/bin/env python
# @Time    : 2021/4/14 9:51
# @Author  : wb
# @File    : toy_dataset.py
import datetime

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
        X, y = make_blobs(n_samples=1800, n_features=2, centers=6, cluster_std=0.6, center_box=(-7, 15))
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
        file_path = '../data/toy_data/Jain.txt'
        with open(file_path, 'r') as f:
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
        file_path = '../data/toy_data/Pathbased.txt'
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            split = line.split()
            X.append([float(split[0]), float(split[1])])
            y.append(int(split[2]))

        return np.array(X), np.array(y)

    def ds_data(self, type='DS4'):
        '''
        DS数据集，4578
        :param type: DS4,DS5,DS7,DS8
        :return:
        '''
        X = []
        # y = []
        if type == 'DS4':
            file_path = '../data/toy_data/t4.8k.dat'
        elif type == 'DS5':
            file_path = '../data/toy_data/t5.8k.dat'
        elif type == 'DS7':
            file_path = '../data/toy_data/t7.10k.dat'
        elif type == 'DS8':
            file_path = '../data/toy_data/t8.8k.dat'
        else:
            file_path = '../data/toy_data/t4.8k.dat'

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            split = line.split()
            X.append([float(split[0]), float(split[1])])

        return np.array(X)

    def neighbors(self, data, n_neighbors):
        '''
        从输入数据中找到邻居点
        :param data: 数据
        :n_neighbors: 邻居数量
        :return: neigh_ind 邻居的ID；neigh_dist 邻居的距离
        '''
        neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=0.4, n_jobs=-1)
        neigh.fit(data)
        neigh_dist, neigh_ind = neigh.kneighbors()

        return neigh_ind, neigh_dist

    def divide_type(self, neigh_ind, n_neighbors, param_lambda_low, param_lambda_high):
        '''
        为所有的无标签样本划分类型，分为主干点，边界点，噪声点
        :param neigh_ind: 邻居表
        :param param_lambda_low: 噪声点与边界点阈值
        :param param_lambda_high: 边界点和主干点的阈值
        :param n_neighbors: 邻居数量K
        :return:
        '''

        starttime = datetime.datetime.now()

        # 主干点
        backbone_point = []
        # 边界点
        border_point = []
        # 噪声点
        noise_point = []

        # r值
        r_list = []
        # 共享邻居数量
        for index in neigh_ind:
            # enumerate neigh_index[0]是id neigh_index[1]是值
            snn_list = []
            # 这个点的邻居列表
            for neighbor in index:
                # 求该数据点的邻居与邻居的邻居有多少是重复邻居
                snn = list(set(index).intersection(set(neigh_ind[neighbor])))
                # 共享邻居的数量
                snn_num = len(snn)
                # 把每个邻居的共享邻居保存起来
                snn_list.append(snn_num)
            # 每个点的平均邻居数
            snn_avg = np.mean(snn_list)
            # 计算r值
            r = snn_avg / n_neighbors
            r_list.append(r)

        print('r均值', np.mean(r_list))
        print('r中位数', np.median(r_list))
        # return r_list

        # 设置para_lambda为均值
        # para_lambda = np.mean(r_list)
        # 划分点，并输出点的id
        for r in enumerate(r_list):
            # 主干点 backbone
            if (r[1] >= param_lambda_high and r[1] <= 1):
                backbone_point.append(r[0])
            elif (r[1] >= param_lambda_low and r[1] <= param_lambda_high):
                border_point.append(r[0])
            elif (r[1] >= 0 and r[1] < param_lambda_low):
                noise_point.append(r[0])
            else:
                print('出错了')

        endtime = datetime.datetime.now()
        print('节点划分类型用时', (endtime - starttime).seconds)

        return backbone_point, border_point, noise_point

    def build_density(self, neigh_ind, neigh_dist):
        '''
        计算每个数据点的密度
        使用SNN的方式，共享邻居数据点
        两个邻居数据点才有相似度，相似度公式为S/d(i)+d(j)，每个数据点的密度是K邻居的相似度之和
        :param neigh_ind: 邻居id
        :param neigh_dist:  邻居的距离
        :return: density 每个点的密度值
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
        :param data: 数据
        :param density: 密度
        :param neigh_dist: 邻居距离
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
                delta = (dij + (neigh_dist_avg[sort_density_idx[node_i]] + neigh_dist_avg[sort_density_idx[node_j]]))
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
        :return: scores 每个节点的得分
        '''
        scores = []
        max_rho = np.max(density)
        max_delta = np.max(interval)

        for rho, delta in zip(density, interval):
            # 每个数据点的得分计算
            score = (rho / max_rho) * (delta / max_delta)
            # score = rho * delta
            scores.append(score)
        return scores

    def detect_jump_point(self, scores, param_alpha):
        '''
        动态选择簇头
        f(x, a, k) = akax−(a + 1)
        logf(x, a, k) = alog(k) + log(a) − (a + 1)log(x)
        本函数全部按照论文中伪代码编写而成
        主要的流程就是，通过阈值找跳变点，因为score排序过，所以找到跳变的k，k前面的就全部是簇头
        :param scores: 数组scores的元素按升序排列
        :param param_alpha: 置信度参数 alpha
        :return: e 跳点e的对应索引
        '''
        # 长度
        n = len(scores)
        # 返回的簇的数量
        e = -1
        # 阈值
        w_n = 0
        # score_index = np.argsort(-np.array(scores))
        # 因为先取反进行降序排序的，所以最后需要取绝对值
        # sorted_scores = abs(np.sort(-np.array(scores)))
        # 论文中需要升序排序
        sorted_scores = np.sort(np.array(scores))
        for k in range(int(n/2), n-3):
            m_a = np.mean(sorted_scores[0:k])
            m_b = np.mean(sorted_scores[k:n])
            if m_a <= param_alpha * m_b:
                # a的参数，shape就是k，scale就是a
                shape_a = sorted_scores[0]
                sum_a = 0
                for i in range(0, k):
                    sum_a += np.log(sorted_scores[i] / shape_a)
                scale_a = k / sum_a
                # b的参数
                shape_b = sorted_scores[k]
                sum_b = 0
                for i in range(k, n):
                    sum_b += np.log(sorted_scores[i] / shape_b)
                scale_b = (n - k + 1) / sum_b
                sk = 0
                for i in range(k, n):
                    ta = scale_a * np.log(shape_a) + np.log(scale_a) - (scale_a + 1) * np.log(sorted_scores[i])
                    tb = scale_b * np.log(shape_b) + np.log(scale_b) - (scale_b + 1) * np.log(sorted_scores[i])
                    sk += np.log(tb / ta)
                if sk > w_n:
                    w_n = sk
                    e = k
        return e

    # def select_head(self, scores, class_num):
    #     '''
    #     根据每个数据点的分数，选择簇头
    #     :param scores: 数据点分数
    #     :param class_num: 类别数
    #     :return: 簇节点的ID heads []
    #     '''
    #
    #     # 降序排序，需要选取分数最大的作为簇头
    #     score_index = np.argsort(-np.array(scores))
    #     # 有多少个故障类别，就有多少个簇头
    #     heads = score_index[:class_num].tolist()
    #
    #     return heads

    # def divide_area(self, density, interval, param_lambda):
    #     '''
    #     划分区域
    #     :param density: 密度
    #     :param interval: 间隔
    #     :param param_lambda: 划分区域的参数lambda
    #     :return: areas [[core_region], [border_region], [new_category_region]]
    #     '''
    #     # 1.在rho和delta的决策图中划分区域
    #     # 2.把所有的无标签点分配到这些区域
    #     # 3.输出每个区域内的数据点ID
    #
    #     # 密度的分割线，平均密度
    #     rho_split_line = np.mean(density)
    #     # 间隔的分割线，lambda*间隔的方差
    #     delta_split_line = param_lambda * np.var(interval)
    #
    #     # 根据分割线划分区域
    #     # 核心区域
    #     core_region = []
    #     # 边缘区域
    #     border_region = []
    #     # 新类别区域
    #     new_category_region = []
    #     # 数据ID
    #     index = 0
    #     for rho, delta in zip(density, interval):
    #
    #         if rho >= rho_split_line:
    #             core_region.append(index)
    #         elif rho < rho_split_line and delta < delta_split_line:
    #             border_region.append(index)
    #         elif rho < rho_split_line and delta >= delta_split_line:
    #             new_category_region.append(index)
    #         else:
    #             print('没这种数据')
    #
    #         index = index + 1
    #
    #     # 最后输出的三个区域的值
    #     areas = [core_region, border_region, new_category_region]
    #
    #     return areas

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
        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y) + 1))))

        # 添加X/Y轴描述
        plt.xlabel('x')
        plt.ylabel('y')

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
        plt.rcParams['savefig.dpi'] = 200  # 图片像素
        plt.rcParams['figure.dpi'] = 200  # 分辨率
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y) + 1))))

        plt.scatter(X[:, 0], X[:, 1], s=3, color=colors[y])
        for head in heads:
            plt.scatter(X[head, 0], X[head, 1], s=10, color='k', marker='*')

        plt.show()

    def plot_rho_delta(self, density, interval, y, heads):
        '''
        绘制rho-delta 密度间隔决策图
        :param density: 密度
        :param interval: 间隔
        :param y: 类别标签，帮助绘制颜色的
        :param heads: 簇头
        :return:
        '''
        plt.rcParams['savefig.dpi'] = 200  # 图片像素
        plt.rcParams['figure.dpi'] = 200  # 分辨率
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

    def plot_scores(self, scores):
        '''
        绘制分数图
        :param scores: 节点的分数
        :return:
        '''
        # sorted_scores = abs(np.sort(-np.array(scores)))
        sorted_scores = np.sort(np.array(scores))
        index = [i for i in range(len(scores))]
        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率

        # 添加X/Y轴描述
        plt.xlabel('n')
        plt.ylabel('rho*delta')
        plt.scatter(index, sorted_scores, s=3)

        plt.show()

    # def plot_pointwithtype(self):
    #     '''
    #     绘制数据点的type
    #     :return:
    #     '''

if __name__ == '__main__':
    toy = ToyDataset()
    # 普通数据集的情况
    X, y = toy.blobs_data()
    # toy.plot_data(X, y)
    # 获取邻居
    n_neighbors = int(len(X) * 0.05)
    neigh_ind, neigh_dist = toy.neighbors(X, n_neighbors)
    # print(neigh_dist)

    # 计算密度与间隔
    density = toy.build_density(neigh_ind, neigh_dist)
    # print(density)
    interval = toy.build_interval(X, density, neigh_dist)
    # print(interval)

    # 计算得分
    scores = toy.score(density, interval)
    # toy.plot_scores(scores)

    # 自动获取聚类中心
    # 在论文SAND中的alpha为0.05.这里的alpha为SAND中的1-alpha，所以设置为0.95

    param_alpha = 0.95
    k = toy.detect_jump_point(scores, param_alpha)

    print('K值', k)

    # # 找到聚类中心
    # heads = toy.select_head(scores, class_num=3)
    # # 绘制聚类中心
    # toy.plot_heads(X, y, heads)
    # # 绘制密度与间隔
    # toy.plot_rho_delta(density, interval, y, heads)

    # '#377eb8', '#ff7f00', '#4daf4a'

    #######################################################################################
    # 无标签数据集
    # X = toy.ds_data(type='DS7')
    # n_neighbors = 50
    # # n_neighbors = int(len(X) * 0.05)
    # # 获取邻居
    # neigh_ind, neigh_dist = toy.neighbors(X, n_neighbors)
    # print(len(neigh_ind))

    # 划分节点类型
    # param_lambda_low = 0.52311
    # para_lambda_high = 0.57111
    # backbone_point, border_point, noise_point = toy.divide_type(neigh_ind, n_neighbors,
    # param_lambda_low, para_lambda_high)
    # print(len(backbone_point), len(border_point), len(noise_point))

    # # 计算密度与间隔
    # density = toy.build_density(neigh_ind, neigh_dist)
    # print(density)
    # interval = toy.build_interval(X, density, neigh_dist)
    # print(interval)

    # # 计算节点的得分
    # scores = toy.score(density, interval)

    # # 自动获取聚类中心
    # param_alpha = 2
    # k = toy.detect_jump_point(scores, param_alpha)
    #
    # print('K值', k)

    # 设置分辨率
    # plt.rcParams['savefig.dpi'] = 300  # 图片像素
    # plt.rcParams['figure.dpi'] = 300  # 分辨率

    # 绘制密度图
    # colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
    #                                      '#f781bf', '#a65628', '#984ea3',
    #                                      '#999999', '#e41a1c', '#dede00']),
    #                               int(max(y) + 1))))
    # # 添加X/Y轴描述
    # plt.xlabel('rho')
    # plt.ylabel('delta')
    #
    # plt.scatter(density, interval, s=3, color=colors[y])
    # plt.show()

    # 绘制节点邻居
    # plt.scatter(X[:, 0], X[:, 1], s=2)
    # plt.scatter(X[0, 0], X[0, 1], s=5, c='#4daf4a', marker='^')
    # for neigh in neigh_ind[0]:
    #     plt.scatter(X[neigh, 0], X[neigh, 1], s=2, c='#ff7f00')

    # 添加X/Y轴描述
    # plt.xlabel('x')
    # plt.ylabel('y')

    # 绘制数据图像
    # plt.scatter(X[:, 0], X[:, 1], s=3)

    # 绘制节点类型
    # for backbone in backbone_point:
    #     plt.scatter(X[backbone, 0], X[backbone, 1], s=3, c='#377eb8')
    # for border in border_point:
    #     plt.scatter(X[border, 0], X[border, 1], s=3, c='#ff7f00')
    # for noise in noise_point:
    #     plt.scatter(X[noise, 0], X[noise, 1], s=3, c='#4daf4a')

    # plt.show()


