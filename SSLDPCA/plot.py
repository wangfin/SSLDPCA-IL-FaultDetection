#!/usr/bin/env python
# @Time    : 2021/3/23 15:52
# @Author  : wb
# @File    : plot.py

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

'''
绘制图形
'''

class Plot(object):

    def plot_data(self, data, label):
        '''
        绘制图形
        '''

        # 点图采用T-SNE

        X_tsne = TSNE(n_components=2, random_state=0).fit_transform(data, label)

        ckpt_dir = "images"
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, cmap='Spectral', label=label)

        plt.savefig('images/data.png', dpi=600)
        plt.show()

    def plot_areas(self, data, areas):
        '''
        绘制区域示意图
        '''
        # 核心区域
        core = []
        for index in areas[0]:
            core.append(data[index])
        core_label = [0 for _ in range(len(core))]
        # 边缘区域
        border = []
        for index in areas[1]:
            border.append(data[index])
        border_label = [1 for _ in range(len(border))]
        # 新类别区域
        new_category = []
        for index in areas[2]:
            new_category.append(data[index])
        new_category_label = [2 for _ in range(len(new_category))]

        # 合并数据
        areas_data = core + border + new_category
        areas_label = core_label + border_label + new_category_label

        areas_sne = TSNE(n_components=2, random_state=0).fit_transform(areas_data, areas_label)

        plt.scatter(areas_sne[:, 0], areas_sne[:, 1], c=areas_label, cmap='Spectral', label=areas_label)

        plt.savefig('images/areas.png', dpi=600)
        plt.show()

    def plot_pseudo_labels(self):
        '''
        绘制伪标签示意图
        '''







