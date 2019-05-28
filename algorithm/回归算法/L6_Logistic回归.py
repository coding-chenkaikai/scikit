# -*- coding: UTF-8 -*-
"""
    Logistic回归预测乳腺癌类别
    算法：Logistic回归
    数据：datas/breast-cancer-wisconsin.data
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
path = "datas/breast-cancer-wisconsin.data"
names = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
         'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
df = pd.read_csv(path, header=None, names=names)
datas = df.replace('?', np.NAN).dropna(axis=0, how='any')

x = datas[names[1:10]]
y = datas[names[10]]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# 模型训练
model = Pipeline([
    ('ss', StandardScaler()),
    ('lr', LogisticRegressionCV(multi_class='ovr', fit_intercept=True, Cs=np.logspace(-2, 2, 20),
                                cv=2, penalty='l2', solver='lbfgs', tol=0.01))
])

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

result = model.get_params()['lr']
print('r：', model.score(x_train, y_train))
print('参数：', result.coef_)
print('截距：', result.intercept_)

t = np.arange(len(x_test))
plt.figure(figsize=(14, 7), facecolor='w')
plt.ylim(0, 6)
plt.plot(t, y_test, 'ro', markersize=8, zorder=3, label=u'真实值')
plt.plot(t, y_predict, 'go', markersize=14, zorder=2, label=u'预测值,$R^2$=%.3f' % result.score(x_test, y_test))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'乳腺癌类型', fontsize=18)
plt.title(u'Logistic回归预测乳腺癌类别', fontsize=20)
plt.show()