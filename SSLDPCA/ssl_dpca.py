#!/usr/bin/env python
# @Time    : 2021/3/9 19:25
# @Author  : wb
# @File    : ssl_dpca.py
import os
import random

from config import opt

'''
半监督（SSL）的密度峰值聚类（DPCA）
'''

class SslDpca(object):
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
    '''
    def __init__(self, root):
        '''
        在其中注入小部分的有标签数据
        :param root: 图像数据的目录
        '''
        # 有标签数据的占比
        self.label_fraction = opt.label_fraction
        # 故障的类别
        self.category = opt.category
        # 全部的图片
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]
        # 图像的ID
        self.category_imgs_index = []

    def label_data_index(self):
        '''
        处理有标签的数据，选择其中的一部分作为标签数据输入算法，其他数据的标签全部清除

        :return: 每个类别的有标签的数据的ID
        '''
        # 选取一定比例的有标签数据（这个是仿真中可以调整的参数）
        # 为了实验起见，选取平衡的数据，即每个类分配相同比例的有标签数据集
        for category in range(self.category):
            category_img_index = []
            # 读取图片的编号
            for index in range(len(self.imgs)):
                # 提取出每一个图像的标签
                label = int(self.imgs[index].split('/')[-1].split('\\')[-1].split('.')[0])
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
        print(input_label_index[0])


if __name__ == '__main__':
    root = os.path.join('../', opt.train_data_root)
    ssldpca = SslDpca(root)
    ssldpca.label_data_index()


