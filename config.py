#!/usr/bin/env python
# @Time    : 2021/3/2 15:44
# @Author  : wb
# @File    : config.py

'''
配置文件，配置项目的超参数
'''

import warnings
import torch as t

class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'ResNet34'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    vis_port = 8097  # visdom 端口

    train_data_root = './data/CWRU_data_2d/DE/gadf'  # 训练集存放路径，测试集从训练集中划出来
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    CWRU_data_root = './CWRU_data'  # CWRU的数据列表
    CWRU_data_1d_root = './CWRU_data_1d'  # CWRU数据1d的保存路径，h5文件
    CWRU_data_2d_root = './CWRU_data_2d'  # CWRU数据2d的根目录保存路径

    CWRU_dim = 400  # CWRU的数据维度
    CWRU_category = 101  # CWRU总共有101个类别

    train_fraction = 0.8  # 训练集数据的占比

    batch_size = 64  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 1  # print info every N batch

    debug_file = './tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = './result/result.csv'

    max_epoch = 10
    lr = 0.1  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数

    # 下面是半监督的部分参数
    label_fraction = 0.2  # 选取有标签样本的占比
    K = 11  # KNN的K值

    def parse(self, kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()
