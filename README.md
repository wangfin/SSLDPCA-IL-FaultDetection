# 基于半监督式增量学习的工业物联网设备故障诊断算法

## 简介

设计了一种基于半监督式增量学习的工业物联网设备故障诊断系统，该系统包括：故障诊断模块、半监督标记模块、增量更新模块。

该系统针对传统数据驱动的故障诊断方法存在的增量更新能力与学习无标签样本数据能力不足的问题做出了改进。在面对数据时变与缺乏标签的情况时，保证故障诊断模型能够有效训练，及时更新，保持较高的故障诊断准确率。

## 整体架构

- 故障诊断模块
- 半监督标记模块
- 增量更新模块

故障诊断模块读取设备监测数据，根据数据判断设备是否处于正常状态，如果出现故障，判断设备发生何种故障；

半监督标记模块首先判断设备监测数据中是否存在未知的故障类别样本，并对所有无标签的设备监测数据（包括已知故障类别与未知故障类别样本）标记伪标签，最后输出带有伪标签的样本以辅助增量更新模块对故障诊断模块进行更新；

增量更新模块使用半监督标记模块输出的伪标签样本对故障诊断模块进行增量地更新。

流程图：

![流程图](https://notes-pic.oss-cn-shanghai.aliyuncs.com/%E6%95%85%E9%9A%9C%E8%AF%8A%E6%96%AD%E6%96%B9%E6%A1%88/%E6%9E%B6%E6%9E%84%E5%9B%BE-%E6%9C%80%E6%96%B0%E7%89%88.png)

## 环境配置

- python3.6
- tslearn 0.5.0.5 `tslearn`是一个Python软件包，提供了用于分析时间序列的机器学习工具。
- scikit-learn 0.23.2 机器学习库
- pytorch 1.7.0 深度学习库
- dcipy 科学计算
- numpy 1.19.2 矩阵计算
- h5py 2.10.0 用来存储使用h5文件
- pandas 1.1.3 存储（好像没用到）
- matplotlib 3.3.2 绘图
- seaborn 0.11.1 绘图
- tqdm 4.54.1 进度条
- xlrd,xlwt 处理表格