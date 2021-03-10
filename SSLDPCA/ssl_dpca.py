#!/usr/bin/env python
# @Time    : 2021/3/9 19:25
# @Author  : wb
# @File    : ssl_dpca.py

'''
半监督（SSL）的密度峰值聚类（DPCA）
'''

class SslDpca(object):
    '''
    半监督的DPCA，在原始的DPCA的基础上加入半监督（小部分有标签数据）
    步骤：
    1.计算密度与间隔
    2.选取
    '''
    def __init__(self):
