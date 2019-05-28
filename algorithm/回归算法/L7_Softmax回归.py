# -*- coding: UTF-8 -*-
"""
    Softmax回归预测葡糖酒质量
    算法：Softmax回归
    数据：datas/winequality-red.csv、datas/winequality-white.csv
"""

import time
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from pandas import DataFrame
from sklearn.externals import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegressionCV

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

## 拦截异常
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# 数据预处理
red_path = 'datas/winequality-red.csv'
red_df = pd.read_csv(red_path, sep=';', low_memory=False)
red_df['type'] = 1

white_path = 'datas/winequality-white.csv'
white_df = pd.read_csv(white_path, sep=';', low_memory=False)
white_df['type'] = 2

df = pd.concat([red_df, white_df], axis=0)
datas = df.replace('?', np.NAN).dropna(axis=0, how='any')

names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
         "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
         "pH", "sulphates", "alcohol", "type"]
x = datas[names]
y = datas['quality']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=0)

# 模型训练
model = Pipeline([
    ('ss', StandardScaler()),
    ('lr', LogisticRegressionCV(multi_class='multinomial', fit_intercept=True, Cs=np.logspace(-5, 1, 100),
                                penalty='l2', solver='lbfgs'))
])

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

result = model.get_params()['lr']
print('r：', model.score(x_train, y_train))
print('参数：', result.coef_)
print('截距：', result.intercept_)

t = np.arange(len(x_test))
plt.figure(figsize=(14, 7), facecolor='w')
plt.ylim(-1, 11)
plt.plot(t, y_test, 'ro', markersize=8, zorder=3, label=u'真实值')
plt.plot(t, y_predict, 'go', markersize=14, zorder=2, label=u'预测值,$R^2$=%.3f' % result.score(x_test, y_test))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'葡糖酒质量', fontsize=18)
plt.title(u'Softmax回归预测葡糖酒质量', fontsize=20)
plt.show()