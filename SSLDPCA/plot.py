#!/usr/bin/env python
# @Time    : 2021/3/23 15:52
# @Author  : wb
# @File    : plot.py

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import itertools
import numpy as np
import openpyxl

'''
绘制图形
'''

class Plot(object):
    def plot_data(self, data, label):
        '''
        绘制图形
        '''

        # 点图采用T-SNE

        X_tsne = TSNE(n_components=2,
                      perplexity=20.0,
                      early_exaggeration=12.0,
                      learning_rate=300.0,
                      init='pca').fit_transform(data, label)

        ckpt_dir = "images"
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, cmap='Spectral', label=label)

        # plt.savefig('images/data.png', dpi=600)
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

    def plot_pseudo_labels(self, data, true_labels, pseudo_labels):
        '''
        绘制伪标签示意图，将原有类别标签与伪标签对比展示
        '''
        plt.figure(dpi=600)
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        true_sne = TSNE(n_components=2, random_state=0).fit_transform(data, true_labels)
        pseudo_sne = TSNE(n_components=2, random_state=0).fit_transform(data, pseudo_labels)

        ax1.scatter(true_sne[:, 0], true_sne[:, 1], c=true_labels, cmap='Spectral', label=true_labels)
        ax2.scatter(pseudo_sne[:, 0], pseudo_sne[:, 1], c=pseudo_labels, cmap='Spectral', label=pseudo_labels)

        plt.show()

    # 绘制混淆矩阵
    def plot_confusion_matrix(self, cm, normalize=False, map='Reds'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Input
        - cm : 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize : True:显示百分比, False:显示个数
        - map :Blues, Greens, Reds
        """
        plt.rcParams['savefig.dpi'] = 300  # 图片像素
        plt.rcParams['figure.dpi'] = 300  # 分辨率
        plt.rcParams["image.cmap"] = map
        plt.rcParams["savefig.bbox"] = 'tight'
        plt.rcParams["savefig.pad_inches"] = 0.2

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        print(cm)
        plt.imshow(cm, interpolation='nearest')
        # plt.title(title)
        # plt.colorbar()
        classes = ['NC', 'IF-1', 'OF-1', 'BF-1', 'IF-2', 'OF-2', 'BF-2', 'IF-3', 'OF-3', 'BF-3']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.1f' # if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=9)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('../pic/con-C.png', transparent=True)
        # plt.show()

    def excel2matrix(self, path, sheet=2):
        '''
        读入excel数据，转换为矩阵
        :param path:
        :param sheet:1
        :return:
        '''
        data = openpyxl.load_workbook(path)
        table = data.worksheets[sheet]
        data = []
        for row in table.iter_rows(min_col=1, max_col=10, min_row=2, max_row=11):
            data.append([cell.value for cell in row])

        datamatrix = np.array(data)
        return datamatrix


if __name__ == '__main__':
    plot = Plot()
    cnf_matrix = plot.excel2matrix(path='../data/matrix.xlsx')
    plot.plot_confusion_matrix(cm=cnf_matrix)



