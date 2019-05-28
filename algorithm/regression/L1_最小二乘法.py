# -*- coding: UTF-8 -*-
"""
    线性回归预测功率与电流之间的关系
    算法：最小二乘法
    数据：datas/household_power_consumption_1000.txt
"""

import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from pandas import DataFrame
from sklearn.model_selection import train_test_split

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

path = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';', low_memory=False)

x = df.iloc[:, 2:4]
y = df.iloc[:, 5]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

X = np.mat(x_train)
Y = np.mat(y_train).reshape(-1, 1)
theta = (X.T * X).I * X.T * Y
y_predict = np.mat(x_test) * theta

t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', lw=2, label=u'真实值')
plt.plot(t, y_predict, 'g-', lw=2, label=u'预测值')
plt.legend(loc='upper left')
plt.title(u'线性回归预测功率与电流之间的关系', fontsize=20)
plt.grid(b=True)
plt.show()


