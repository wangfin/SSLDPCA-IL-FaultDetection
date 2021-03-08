#!/usr/bin/env python
# @Time    : 2021/3/8 15:06
# @Author  : wb
# @File    : main.py

'''
main文件，用于训练，测试等
'''
import torch
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm

import models
from config import opt
from utils.visualize import Visualizer
from data.dataset import CWRUDataset


def train(**kwargs):
    '''
    训练
    :param kwargs: 可调整参数，默认是config中的默认参数
    :return:训练出完整模型
    '''

    # 根据命令行参数更新配置
    opt._parse(kwargs)
    # visdom绘图程序
    vis = Visualizer(opt.env, port=opt.vis_port)

    # step:1 构建模型
    # 选取配置中名字为model的模型
    model = getattr(models, opt.model)()
    # 是否读取保存好的模型参数
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # step2: 数据
    train_data = CWRUDataset(opt.train_data_root, train=True)
    val_data = CWRUDataset(opt.val_data_root, train=False)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False)

    # step3: 目标函数和优化器
    # 损失函数，交叉熵
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    # 优化函数，Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)
    # 学习率衰减
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_decay_iters, opt.lr_decay)

    # step4: 统计指标，平滑处理之后的损失，还有混淆矩阵
    # 损失进行取平均及方差计算。
    loss_meter = meter.AverageValueMeter()
    # 混淆矩阵
    confusion_matrix = meter.ConfusionMeter()
    previous_loss = 1e10

    # 训练
    for epoch in range(opt.max_epoch):

        # 重置
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):

            # train model
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            # detach 一下更安全保险
            confusion_matrix.add(score.detach(), target.detach())

            if (ii + 1) % opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])

                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()