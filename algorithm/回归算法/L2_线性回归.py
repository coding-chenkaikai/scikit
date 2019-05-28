# -*- coding: UTF-8 -*-
"""
    线性回归预测时间与功率之间的关系
    算法：线性回归
    数据：datas/household_power_consumption_1000.txt
"""

import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from pandas import DataFrame
from sklearn.externals import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 格式化时间
def date_format(dt):
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 数据预处理
path = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';', low_memory=False)
datas = df.replace('?', np.NAN).dropna(axis=0, how='any')

x = datas.iloc[:, 0:2]
x = x.apply(lambda m : pd.Series(date_format(m)), axis=1)
y = datas['Global_active_power']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 数据标准化
# print(x_train.describe())
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 模型训练
lr = LinearRegression(fit_intercept=True)
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)

print('训练集：', lr.score(x_train, y_train))
print('测试集：', lr.score(x_test, y_test))

mse = np.average((y_predict - y_test) ** 2)
print('mse：', mse)
print('rmse：', np.sqrt(mse))
print('系数：', lr.coef_)
print('截距：', lr.intercept_)

# 模型持久化
# joblib.dump(ss, 'model/data_ss.model')
# joblib.dump(lr, 'model/data_lr.model')
#
# ss = joblib.load('model/data_ss.model')
# lr = joblib.load('model/data_lr.model')
#
# data = [[2006, 12, 17, 12, 25, 0]]
# data = ss.transform(data)
# predict = lr.predict(data)

# 画图
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', lw=2, label=u'真实值')
plt.plot(t, y_predict, 'g-', lw=2, label=u'预测值')
plt.legend(loc='upper left')
plt.title(u'线性回归预测时间与功率之间的关系', fontsize=20)
plt.grid(b=True)
plt.show()

