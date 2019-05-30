# -*- coding: UTF-8 -*-
"""
    numpy序列示例
"""

import numpy as np

# .ndim 维度
# .shape 形状
# .size 元素个数
# .dtype 类型
# .itemsize 元素字节大小

# np.ones(shape=[2, 3])
# np.zeros(shape=[2, 3])
# np.full(shape=[2, 3], fill_value=10)
# np.eye(10) 产生单位矩阵
# np.random.rand(10) 产生10个[0.0,1.0)之间的随机浮点数
# np.random.randn(2, 3) 产生一个形状为[2,3]具有标准正态分布的样本
# np.random.randint(low=1, high=100, size=10) 产生半开区间[low,high)随机整数
# np.arange(start=0, stop=100, step=2) 产生半开区间[start,stop)步长step的序列
# np.linspace(start=1, stop=10, num=10) 产生等差数列
# np.logspace(start=1, stop=100, num=10) 产生等比数列

x = np.random.randint(100, 200, size=(20, ))
print(np.random.choice(x, (2, 3), replace=False))
