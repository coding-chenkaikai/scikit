# -*- coding: UTF-8 -*-
"""
    线性回归预测时间与电压之间的多项式关系
    算法：线性回归
    数据：datas/household_power_consumption_1000.txt
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

# 格式化时间
def date_format(dt):
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

## 拦截异常
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# 数据预处理
path = 'datas/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';', low_memory=False)
datas = df.replace('?', np.NAN).dropna(axis=0, how='any')

x = datas.iloc[:, 0:2]
x = x.apply(lambda m : pd.Series(date_format(m)), axis=1)
y = datas['Voltage']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = Pipeline([
    ('ss', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('lr', LinearRegression(fit_intercept=True))
    # ('lr', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))
    # ('lr', LassoCV(alphas=np.logspace(0, 1, 10), fit_intercept=False))
    # ('lr', ElasticNetCV(alphas=np.logspace(0, 1, 10), l1_ratio=[.1, .5, .7, .9, .95, 1], fit_intercept=False))
])

model.set_params(poly__degree=2).fit(x_train, y_train)
y_predict = model.predict(x_test)
print('系数：', model.get_params()['lr'])

# 画图
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', lw=2, label=u'真实值')
plt.plot(t, y_predict, 'g-', lw=2, label=u'预测值')
plt.legend(loc='upper left')
plt.title(u'线性回归预测时间与电压之间的多项式关系', fontsize=20)
plt.grid(b=True)
plt.show()