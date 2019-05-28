# -*- coding: UTF-8 -*-
"""
    线性回归预测葡糖酒质量
    算法：线性回归
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV

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

models = [
    Pipeline([
            ('Poly', PolynomialFeatures()),
            ('Linear', LinearRegression())
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures()),
            ('Linear', RidgeCV(alphas=np.logspace(-4, 2, 20)))
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures()),
            ('Linear', LassoCV(alphas=np.logspace(-4, 2, 20)))
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures()),
            ('Linear', ElasticNetCV(alphas=np.logspace(-4, 2, 20), l1_ratio=np.linspace(0, 1, 5)))
        ])
]

l = np.arange(len(x_test))
pool = np.arange(1, 4, 1)
colors = []
for c in np.linspace(5570560, 255, len(pool)):
    colors.append('#%06x' % int(c))

plt.figure(figsize=(16, 8), facecolor='w')
titles = u'线性回归预测', u'Ridge回归预测', u'Lasso回归预测', u'ElasticNet预测'

for t in range(4):
    plt.subplot(2, 2, t + 1)
    model = models[t]
    plt.plot(l, y_test, c='r', lw=2, alpha=0.75, zorder=10, label=u'真实值')
    for i, d in enumerate(pool):
        model.set_params(Poly__degree=d)
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        r = model.score(x_train, y_train)
        plt.plot(l, y_predict, c=colors[i], lw=2, alpha=0.75, zorder=i, label=u'%d阶预测值,$R^2$=%.3f' % (d, r))
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title(titles[t], fontsize=18)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
plt.suptitle(u'线性回归预测葡糖酒质量', fontsize=22)
plt.show()
