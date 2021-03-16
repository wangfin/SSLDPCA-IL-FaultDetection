#!/usr/bin/env python
# @Time    : 2020/7/8 15:47
# @Author  : wb
# @File    : cnn1d.py

from torch import nn
from .basic_module import BasicModule

'''
1D的CNN，用于处理1d的时序信号
可以用于构建基础的故障诊断（分类）模块
'''

class Flatten(nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class cnn1d(BasicModule):
    '''
    1dCNN，用于处理时序信号
    '''

    def __init__(self, kernel1=27, kernel2=36, kernel_size=10, pad=0, ms1=4, ms2=4):
        super(cnn1d, self).__init__()
        self.model_name = 'cnn1d'

        self.conv = nn.Sequential(
            nn.Conv1d(1, kernel1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel1),
            nn.ReLU(),
            nn.MaxPool1d(ms1),
            nn.Conv1d(kernel1, kernel1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel1, kernel1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel1),
            nn.ReLU(),
            nn.MaxPool1d(ms2),
            nn.Conv1d(kernel1, kernel2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel2, kernel2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel2),
            nn.ReLU(),
            Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(36, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x