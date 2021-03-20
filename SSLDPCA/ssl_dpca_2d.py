#!/usr/bin/env python
# @Time    : 2021/3/9 19:25
# @Author  : wb
# @File    : ssl_dpca_2d.py
import os
import random
import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist
from tqdm import tqdm
import itertools
import h5py

from config import opt

'''
半监督（SSL）的密度峰值聚类（DPCA），此文件用于2d数据
'''

class SslDpca2D(object):
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
    def __init__(self, root):
        '''
        初始化，读取图片
        :param root: 图像数据的目录
        '''
        # 有标签数据的占比
        self.label_fraction = opt.label_fraction
        # 故障的类别
        self.category = opt.CWRU_category
        # 邻居数量
        self.neighbor = opt.K
        # 全部的图片的ID
        self.imgs_path = [os.path.join(root, img) for img in os.listdir(root)]
        # 图像的ID
        self.category_imgs_index = []

    def label_data_index(self):
        '''
        处理有标签的数据，选择其中的一部分作为标签数据输入算法，其他数据的标签全部清除

        :return: input_label_index，每个类别的有标签的数据的ID集合
        '''
        # 选取一定比例的有标签数据（这个是仿真中可以调整的参数）
        # 为了实验起见，选取平衡的数据，即每个类分配相同比例的有标签数据集
        for category in range(self.category):
            category_img_index = []
            # 读取图片的编号
            for index in range(len(self.imgs_path)):
                # 提取出每一个图像的标签
                label = int(self.imgs_path[index].split('/')[-1].split('\\')[-1].split('.')[0])
                if label == category:
                    category_img_index.append(index)
            # 将每个类别的图片分别保存
            self.category_imgs_index.append(category_img_index)

        input_label_index = []
        # 选取每个类中一定比例的数据作为有标签数据
        for category_index in self.category_imgs_index:
            category_label_index = []
            for _ in range(int(len(category_index)*self.label_fraction)):
                category_label_index.append(category_index[random.randint(0, len(category_index)-1)])
            input_label_index.append(category_label_index)
        return input_label_index

    def local_density(self):
        '''
        计算数据点的密度
        :return:
        '''
        share_neighbor = []

    def build_distance(self):
        '''
        根据图片之间的相互距离，选出每个数据点的K邻居列表，计算出K邻居平均距离
        :return:
        '''
        node_K_neighbor = []

        # 两两组合
        img_iter_path = itertools.combinations(self.imgs_path, 2)
        for node_i_path, node_j_path in img_iter_path:
            node_i = Image.open(node_i_path)
            node_j = Image.open(node_j_path)
            # 两个数据点之间的距离
            distance = self.euclidean_distance(node_i, node_j)
        # 记录每个数据点与其他数据点的距离

        # 从中选出最近的K个，就是K邻居

        # 计算K邻居的平均距离

        return node_K_neighbor

    def build_distance_all(self):
        '''
        上面那个函数主要是进行了组合，减少了需要计算的数量，增加了工作量
        但是据观察发现，其实大部分的时间都是花费在了读取图片的工作上，所以这个是全部读取的函数

        根据图片之间的相互距离，选出每个数据点的K邻居列表，计算出K邻居平均距离
        :return:
        '''

        # 这边我的理解出了一点问题，我完全可以把每个图片读取进来，然后在进行计算
        # 而不是重复的读取，这样浪费了很多时间

        # 所有的图片
        imgs = []
        # 数据点的K邻居的路径
        node_K_neighbor_path = []
        for path in tqdm(self.imgs_path):
            img = Image.open(path)
            imgs.append(np.asarray(img, dtype='uint8'))
            img.close()

        # <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=400x400 at 0x1AD882E8160>
        # 开始计算
        for node_i in tqdm(imgs):
            # 数据点间的距离合集
            node_distance = np.empty(len(self.imgs_path))
            for node_j in imgs:
                # 计算两个图像之间的距离
                distance = self.euclidean_distance(node_i, node_j)
                np.append(node_distance, distance)

            # 排序
            order_node_distance = np.argsort(node_distance)
            # 选取其中K个
            neighbor = order_node_distance[:self.neighbor]
            # 邻居的路径
            neighbor_path = []
            # 保存所有数据点邻居的K邻居
            for nei in neighbor:
                neighbor_path.append(self.imgs_path[nei])
                node_K_neighbor_path.append(neighbor_path)

        f = open('neighbor.txt', 'w')  # output.txt - 文件名称及格式 w - writing
        # 以这种模式打开文件,原来文件内容会被新写入的内容覆盖,如文件不存在会自动创建
        for i in range(len(self.imgs_path)):
            f.write(self.imgs_path[i])
            f.write(':')
            for j in node_K_neighbor_path[i]:
                f.write(j)
                f.write('||')
            f.write('\n')
        f.close()

        return node_K_neighbor

    def euclidean_distance(self, node_i, node_j):
        '''
        计算两个数据点之间的欧几里得距离
        :param node_i: 输入图片的image对象
        :param node_j: 输入图片
        :return: distance，距离
        '''
        # 先对image对象进行ndarry化，然后展平
        node_i = node_i.flatten()
        node_j = node_j.flatten()

        # 统一大小
        # img_j = img_j.resize(img_i.size)

        # 计算距离
        X = np.vstack([node_i, node_j])
        # 距离的值有点太精确了
        distance = pdist(X, 'euclidean')[0]

        return distance







if __name__ == '__main__':
    root = os.path.join('../', opt.train_data_root)
    ssldpca = SslDpca2D(root)
    node_K_neighbor = ssldpca.build_distance_all()
    print(node_K_neighbor)


