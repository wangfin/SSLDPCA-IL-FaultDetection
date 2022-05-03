# !/usr/bin/env python
# -*-coding:utf-8 -*-
'''
# Time       ：2022/3/15 16:42
# Author     ：wb
# File       : autoencoder.py
'''
from torch import nn
from .basic_module import BasicModule

class Flatten(nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# 自动编码器
class autoencoder(BasicModule):
    '''
    自动编码器
    '''
    def __init__(self, kernel1=27, kernel2=36, kernel_size=10, pad=0, ms1=4, ms2=4):
        super(autoencoder, self).__init__()
        self.model_name = 'autoencoder'

        # 输入 [batch size, channels, length] [N, 1, L]
        # 编码器 卷积层
        self.conv = nn.Sequential(
            nn.Conv1d(1, kernel1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel1),
            nn.ReLU(),
            nn.MaxPool1d(ms1),
            nn.Conv1d(kernel1, kernel1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(kernel1, kernel2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel2),
            nn.ReLU(),
            nn.MaxPool1d(ms2),
            nn.Conv1d(kernel2, kernel2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel2),
            nn.ReLU(),
            nn.Dropout(),
        )

        # 解码器 反卷积
        self.transconv = nn.Sequential(
            nn.ConvTranspose1d(kernel2, kernel2, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel2),
            nn.ReLU(),
            nn.MaxUnpool1d(ms2),
            nn.ConvTranspose1d(kernel2, kernel1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel1),
            nn.ReLU(),
            nn.Dropout(),
            nn.ConvTranspose1d(kernel1, kernel1, kernel_size, padding=pad),
            nn.BatchNorm1d(kernel1),
            nn.ReLU(),
            nn.MaxUnpool1d(ms1),
            nn.ConvTranspose1d(kernel1, 1, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.transconv(x)
        return x


