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
from utils import plot

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
       重新修改方案，主要改动：
       1.首先对注入的数据进行类型划分，划分为主干点，边界点，噪声点
       2.保留主干点和边界点，删除噪声点
       3.修改间隔定义公式，现在间隔定义公式与有标签样本无关
       4.在计算完每个节点的分数之后，采用动态选择簇头（不太好用，还是直接赋一个固定值作为类别数）

    '''

    def __init__(self):
        '''
        读取处理好的1d数据文件
        ../data/CWRU_data_1d/CWRU_DE.h5
        设定一些全局变量作为参数
        '''
        # h5文件路径
        file_path = '../data/CWRU_data_1d/CWRU_mini_0_DE.h5'
        # 读取数据
        f = h5py.File(file_path, 'r')
        # 数据，取值，可以用f['data'].value，不过包自己推荐使用f['data'][()]这种方式
        self.data = f['data'][()]
        # 标签
        self.label = f['label'][()]
        # 每个类别的数据块数量
        self.data_num = f['data_num'][()]
        # 数据中每个数据点的SNN数量
        # 这里可以减少一次计算SNN的计算量，不过暂时还没用上
        self.snn_num = []

        # 有标签数据的占比
        self.label_fraction = opt.label_fraction
        # 有标签数据保存的文件
        self.labeled_data_file = './labeled_data.npy'
        self.label_file = './label.npy'
        # 故障的类别
        self.category = opt.CWRU_category
        # 邻居数量，因为计算的K邻居的第一个是自己，所以需要+1
        self.neighbor_num = opt.K + 1
        # 数据的维度
        self.dim = opt.CWRU_dim
        # K邻居模型保存路径
        self.K_neighbor = './K-neighbor_mini.h5'

    def make_labeled_data(self):
        '''
        处理有标签的数据，选择其中的一部分作为标签数据输入算法，其他数据的标签全部清除
        这里选择数据是随机选择的
        所以每次运行此函数得到的有标签样本是变化的
        考虑保存选出的数据，因为原始数据中需要删除这一部分的数据
        :return: labeled_datas，每个类别的有标签的数据集合 [[],[],...,[]]
        '''
        # 选取一定比例的有标签数据（这个是仿真中可以调整的参数）
        # 为了实验起见，选取平衡的数据，即每个类分配相同比例的有标签数据集
        # (33693,400)(33693)
        # 各个类别的数据
        category_data = []
        # 各个类别的标签
        category_label = []
        # 有标签的数据
        labeled_datas = []
        # 有标签的标签
        labels = []

        # 把每个类别的数据切分出来
        point = 0
        for i in range(len(self.data_num)):
            data = self.data[point:point + self.data_num[i]]
            label = self.label[point:point + self.data_num[i]]

            category_data.append(data)
            category_label.append(label)

            point = point + self.data_num[i]

        # 选出有标签的index
        for data, label in tqdm(zip(category_data, category_label)):
            # 有标签的数量
            label_data_num = int(len(data) * self.label_fraction)
            # 对category的格式为(609,400)
            # 随机从数据中取出需要数量的值，需要先转换为list
            data_ = random.sample(data.tolist(), label_data_num)
            label_ = random.sample(label.tolist(), label_data_num)

            # label_data为list，每个list是(400,)的数据
            # 再把list转换为ndarray
            labeled_datas.append(np.array(data_))
            # labeled_data为list，list中为[(121,400),(xxx,400)...]
            labels.append(np.array(label_))

        # # 保存为h5文件
        # f = h5py.File(self.labeled_data_file, 'w')  # 创建一个h5文件，文件指针是f
        # f['labeled_data'] = np.array(labeled_data)  # 将数据写入文件的主键data下面
        # f.close()  # 关闭文件

        # 使用np的保存，不过保存下来是ndarray
        np.save(self.labeled_data_file, labeled_datas)
        np.save(self.label_file, labels)
        return labeled_datas

    def del_labeled_data(self):
        '''
        从self.data，也就是原始数据中删除有标签的数据
        :return:
        '''
        # 从文件中读取保存好的有标签的样本
        # 读取出来是ndarray，需要转换成list
        labeled_datas = np.load(self.labeled_data_file, allow_pickle=True).tolist()
        labels = np.load(self.label_file, allow_pickle=True).tolist()
        # 为了从self.data中删除元素，需要先把ndarray转换为list
        data_list = self.data.tolist()
        label_list = self.label.tolist()
        # 读取每个类别的有标签样本
        for data, label in zip(labeled_datas, labels):
            # 遍历
            for category_data, category_label in zip(data, label):
                # 根据样本值从原始数据中删除样本
                # 使用np的ndarray删除
                data_list.remove(category_data.tolist())
                label_list.remove(category_label.tolist())
        # 最后还得把data转化为ndarray
        self.data = np.array(data_list)
        self.label = np.array(label_list)

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
        对输入的data进行计算邻居与距离
        :param data: 输入数据，data形式为[[],[],...,[]]
        :return: neigh_dist，邻居之间的距离；neigh_ind，邻居的ID
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
        neigh_dist, neigh_ind = knn_model.kneighbors(test_time_series, return_distance=True)

        endtime = datetime.datetime.now()
        print('计算邻居用时', (endtime - starttime).seconds)
        return neigh_dist, neigh_ind

    def divide_type(self, neigh_ind, param_lambda_low, param_lambda_high):
        '''
        获得每个点的邻居列表之后即可以为所有的无标签样本划分类型，分为主干点，边界点，噪声点
        :param neigh_ind: 邻居表
        :param param_lambda_low: 噪声点与边界点阈值
        :param param_lambda_high: 边界点和主干点的阈值
        :return:backbone_point, border_point, noise_point
        主干点，边界点，噪声点的ID值，在data中的index值
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
                # 求该数据点的邻居与其邻居的邻居有多少是重复邻居
                snn = list(set(index).intersection(set(neigh_ind[neighbor])))
                # 共享邻居的数量
                snn_num = len(snn)
                # 把每个邻居的共享邻居保存起来
                snn_list.append(snn_num)
            # 每个点的平均邻居数
            snn_avg = np.mean(snn_list)
            # 计算r值
            # 这里没有使用self.neighbor_num，因为这个数值+1了
            r = snn_avg / opt.K
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
            # 边界点
            elif (r[1] >= param_lambda_low and r[1] <= param_lambda_high):
                border_point.append(r[0])
            # 噪声点
            elif (r[1] >= 0 and r[1] < param_lambda_low):
                noise_point.append(r[0])
            else:
                print('出错了')

        endtime = datetime.datetime.now()
        print('节点划分类型用时', (endtime - starttime).seconds)

        return backbone_point, border_point, noise_point

    def del_noise(self, noise_point, neigh_dist, neigh_ind):
        '''
        从self.data，也就是原始数据中删除noise_point
        :param noise_point: 噪声点的idx
        :return:
        '''
        # 从self.data,neigh_dist,neigh_ind中删除noise_point
        for noise_node in noise_point:
            # np 删除行
            np.delete(self.data, noise_node, axis=0)
            # python list 删除，使用pop
            neigh_dist.pop(noise_node)
            neigh_ind.pop(noise_node)

        return neigh_dist, neigh_ind

    def build_density(self, neigh_dist, neigh_ind):
        '''
        计算每个数据点的密度
        使用SNN的方式，共享邻居数据点
        两个邻居数据点才有相似度，相似度公式为S/d(i)+d(j)，每个数据点的密度是K邻居的相似度之和
        dis和nei的第一个点都是自己本身，需要去掉
        :param: distance,[[],[],[]]数据点的距离
        :param: neighbors_index,[[],[],[]]数据点的邻居列表
        :return: density，密度列表，[, ,...,]
        '''
        starttime = datetime.datetime.now()
        # 每个数据点的密度
        density = []
        for index in range(len(neigh_dist)):
            # 该数据点的平均邻居距离，去掉第一个点，第一个是本身数据点
            node_distance_avg = np.mean(neigh_dist[index][1:])

            # 数据点的密度
            node_density = 0

            # 从一个数据点的邻居内开始计算，neighbor是邻居的ID
            for neighbor in neigh_ind[index]:
                # 求该数据点的邻居与其邻居的邻居有多少是重复邻居
                snn = list(set(neigh_ind[index][1:]).intersection(set(neigh_ind[neighbor][1:])))
                # 共享邻居的数量
                snn_num = len(snn)
                # 邻居数据点的平均距离
                neighbors_distance_avg = np.mean(neigh_dist[neighbor][1:])

                # 两个数据点的相似度
                sim = snn_num / (node_distance_avg + neighbors_distance_avg)
                # 数据点的密度是每个邻居的相似度的和
                node_density += sim

            # 所有数据点的密度
            density.append(node_density)

        endtime = datetime.datetime.now()
        print('计算密度用时', (endtime - starttime).seconds)

        return density

    def build_interval(self, density, neigh_dist):
        '''
        这个函数是与有标签样本无关的版本，目前函数修改为无有标签样本
        :param density: 密度
        :param neigh_dist: 邻居之间的距离
        :return: interval，间隔列表 []
        '''
        # 1.首先需要寻找到比数据点密度更高的数据点
        # 2.然后计算dij，i的平均邻居距离，j的平均邻居距离
        # 3.密度最大值的数据点需要成为最大的间隔值

        starttime = datetime.datetime.now()

        # 数据点的间隔值
        interval = []
        # 因为排序过，所以得换一种dict
        interval_dict = {}

        # 排序，获得排序的ID[]
        sort_density_idx = np.argsort(density)
        # 数据点node的index
        for node_i in range(len(sort_density_idx)):

            # node_i的平均邻居距离
            node_i_distance_avg = np.mean(neigh_dist[sort_density_idx[node_i]])

            # 数据点的全部间隔
            node_intervals = []
            # 密度比node更大的数据点
            for node_j in range(node_i + 1, len(sort_density_idx)):
                # i，j的距离
                dij = self.euclidean_distance(self.data[sort_density_idx[node_i]], self.data[sort_density_idx[node_j]])
                # 数据点j的平均邻居距离
                node_j_distance_avg = np.mean(neigh_dist[sort_density_idx[node_j]])
                delta = dij * (node_i_distance_avg + node_j_distance_avg)
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
        :param data1: 数据1
        :param data2: 数据2
        :return: 距离
        '''

        X = np.vstack([data1, data2])
        distance = pdist(X, 'euclidean')[0]
        return distance

    def build_score(self, density, interval):
        '''
        根据数据点的密度与间隔，计算分数
        :param density: 数据点密度列表 []
        :param interval: 数据点间隔列表 []
        :return: node_scores [] 数据点的分数
        '''

        starttime = datetime.datetime.now()

        node_scores = []
        max_rho = np.max(density)
        max_delta = np.max(interval)

        for rho, delta in zip(density, interval):
            # 每个数据点的得分计算
            score = (rho / max_rho) * (delta / max_delta)
            node_scores.append(score)

        endtime = datetime.datetime.now()
        print('计算得分用时', (endtime - starttime).seconds)

        return node_scores

    def detect_jump_point(self, node_scores, param_alpha):
        '''
        动态选择簇头
        f(x, a, k) = akax−(a + 1)
        logf(x, a, k) = alog(k) + log(a) − (a + 1)log(x)
        本函数全部按照论文中伪代码编写而成
        主要的流程就是，通过阈值找跳变点，因为score排序过，所以找到跳变的k，k前面的就全部是簇头
        不过没有理解论文中的操作，可能是代码有问题，可能是参数设置的问题，反正这玩意不好用
        最后还是直接设置给定值的类别数
        :param node_scores: 数组node_score的元素按升序排列
        :param param_alpha: 置信度参数 alpha
        :return: e 跳点e的对应索引
        '''
        # 长度
        n = len(node_scores)
        # 返回的簇的数量
        e = -1
        # 阈值
        w_n = 0
        # score_index = np.argsort(-np.array(scores))
        # 因为先取反进行降序排序的，所以最后需要取绝对值
        # sorted_scores = abs(np.sort(-np.array(scores)))
        # 论文中需要升序排序
        sorted_scores = np.sort(np.array(node_scores))
        for k in range(int(n / 2), n - 3):
            m_a = np.mean(sorted_scores[0:k])
            m_b = np.mean(sorted_scores[k:n])
            if m_a < param_alpha * m_b:
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

    def select_head(self, node_scores):
        '''
        根据每个数据点的分数，选择簇头
        本来是应该使用跳变点动态选择类别的，不过还是用这个混混吧
        :param node_scores: 数据点分数
        :return: 簇节点的ID heads []
        '''
        starttime = datetime.datetime.now()

        # 降序排序，需要选取分数最大的作为簇头
        score_index = np.argsort(-np.array(node_scores))
        # 有多少个故障类别，就有多少个簇头
        head_nodes = score_index[:self.category].tolist()

        endtime = datetime.datetime.now()
        print('计算簇头用时', (endtime - starttime).seconds)

        return head_nodes

    def assign_labels(self, head_nodes, type_point, labeled_data):
        '''
        为无标签样本标注伪标签，也就是对聚类中心，主干点，边界点分别标注标签
        聚类中心：哪个已知类别的真实标签样本与聚类中心的平均距离最近，那么聚类中心的标签就是该已知类的标签
        主干点：主干点分配给距离最近的聚类中心，也就是与聚类中心保持一致
        边界点：边界点与距离他最近的K个主干点的标签值保持一致
        :param head_nodes: 簇中心点
        :param type_point: 不同区域的数据ID [[backbone_point],[border_point]]
        :param labeled_data: 有标签样本
        :return: 样本的伪标签值 []
        '''
        starttime = datetime.datetime.now()

        # 主干点
        backbone_point = type_point[0]
        # 边界点
        border_point = type_point[1]

        # 簇中心点的标签
        heads_labels = []
        # 簇中心点分配标签的过程
        for head_node in head_nodes:
            # 计算数据点node到有标签样本的平均距离，然后取最小的平均类别距离
            label_dis = []
            # 每个类别的有标签样本
            for category in labeled_data:
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

        # 主干点的标签分配
        backbone_labels = []
        for backbone_node in backbone_point:
            head_dis = []
            for head_node in head_nodes:
                # 主干点与簇中心点的距离
                dis = self.euclidean_distance(self.data[backbone_node], self.data[head_node])
                head_dis.append(dis)
            # 核心区域点的标签值与最近的簇中心点保持一致
            backbone_label = heads_labels[int(np.argmin(head_dis))]
            backbone_labels.append(backbone_label)

        # 边缘区域的标签分配
        border_labels = []
        for border_node in border_point:
            # 计算距离
            border_node_dis = []
            for backbone_node in backbone_point:
                # 边缘区域中的点与核心区域内点的距离
                dis = self.euclidean_distance(self.data[border_node], self.data[backbone_node])
                border_node_dis.append(dis)
            # 保存K邻居的标签值
            K_labels = []
            # 找到距离边缘点最近的核心点
            for i in np.argsort(border_node_dis)[:opt.K]:
                K_labels.append(backbone_labels[i])

            # 这里是dict，Counter({3: 2, 10: 2, 1: 1, 0: 1})
            max_K_labels = Counter(K_labels)
            # 按value对dict排序，逆序排序
            max_K_labels = sorted(max_K_labels.items(), key=lambda item: item[1], reverse=True)
            # max_K_labels[0]为最大值，max_K_labels[0][0]为最大值的key
            border_labels.append(max_K_labels[0][0])

        # 新类别区域的标签分配
        # new_category_labels = []
        # new_category_labels = self.new_category_label(new_category_region)

        # 最后需要把标签按顺序摆好，然后输出
        pseudo_labels = []
        # 把几个list合并一下
        data_index = head_nodes + backbone_point + border_point
        data_labels = heads_labels + backbone_labels + border_labels
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

if __name__ == '__main__':
    ssldpca = SslDpca1D()

    # 绘制原始数据的t-sne图
    plot = plot.Plot()
    plot.plot_data(ssldpca.data, ssldpca.label)
    # # 选出有标签的数据，准备注入
    # labeled_data = ssldpca.make_labeled_data()
    # # 删除有标签数据
    # ssldpca.del_labeled_data()
    # # # 构建邻居模型
    # # ssldpca.neighbors_model()
    # # 计算邻居，取得邻居距离和邻居idx
    # neigh_dist, neigh_ind = ssldpca.neighbors()
    # # 给所有节点划分类型
    # param_lambda_low = 0.52311
    # para_lambda_high = 0.57111
    # # 三种类型，backbon_point 主干点;border_point 边界点;noise_point 噪声点
    # backbone_point, border_point, noise_point = ssldpca.divide_type(neigh_ind, param_lambda_low, para_lambda_high)
    # print(len(backbone_point), len(border_point), len(noise_point))
    #
    # # 删除噪声点，self.data,neigh_dist,neigh_ind,都删除
    # neigh_dist, neigh_ind = ssldpca.del_noise(noise_point, neigh_dist, neigh_ind)
    # # 计算密度
    # density = ssldpca.build_density(neigh_dist, neigh_ind)
    # # 计算间隔
    # interval = ssldpca.build_interval(density, neigh_dist)
    #
    # # 计算节点分数
    # node_scores = ssldpca.build_score(density, interval)
    # head_nodes = ssldpca.select_head(node_scores)
    #
    # # 获取数据的伪标签
    # pseudo_labels = ssldpca.assign_labels(head_nodes, [backbone_point, border_point], labeled_data)
    # # print(pseudo_labels)





