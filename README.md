# 基于半监督式增量学习的工业物联网设备故障诊断方法研究

## 简介

设计了一种基于半监督式增量学习的工业物联网设备故障诊断系统，该系统包括：故障诊断模块、半监督标记模块、增量更新模块。该系统针对传统数据驱动的故障诊断方法存在的增量更新能力与学习无标签样本数据能力不足的问题做出了改进。在面对数据时刻变化与数据缺乏标签的情况时，保证故障诊断模型能够及时更新，并且在样本标签缺失的条件下有效训练，保持较高的故障诊断准确率。

## 模块

- 故障诊断模块
- 半监督标记模块
- 增量更新模块

故障诊断模块读取设备监测数据，根据数据判断设备是否处于正常状态，如果出现故障，判断设备发生何种故障；

半监督标记模块作用是辅助增量更新模块，该模块首先判断设备监测数据中是否存在未知的故障类别样本数据，在得出结果后对未知类别与已知类别的样本数据进行不同的处理，最后使用少量带有真实标签的样本数据对所有无标签的样本数据标记伪标签，并输出带有伪标签的样本数据；

所述增量更新模块使用半监督标记模块输出的伪标签样本数据对故障诊断模块进行增量地更新。



## 环境配置

- tslearn 0.5.0.5 `tslearn`是一个Python软件包，提供了用于分析时间序列的机器学习工具。
- 