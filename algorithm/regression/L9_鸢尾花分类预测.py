# -*- coding: UTF-8 -*-
"""
    Logistic回归对比KNN预测鸢尾花类别
    算法：Logistic回归
    数据：datas/iris.data
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt
import sklearn.preprocessing as pre_processing

from pandas import DataFrame
from sklearn.externals import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegressionCV

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

## 拦截异常
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# 数据预处理
path = "datas/iris.data"
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'cla']
df = pd.read_csv(path, header=None, names=names)
df = df.replace('?', np.NAN).dropna(axis=0, how='any')
# print(df['cla'].value_counts())

datas = df.iloc[:, :-1]
label = pre_processing.LabelEncoder()
datas['cla'] = label.fit_transform(df.iloc[:, -1])

x = datas.iloc[:, :-1]
y = datas.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Logistic模型训练
model = Pipeline([
    ('ss', StandardScaler()),
    ('lr', LogisticRegressionCV(multi_class='multinomial', fit_intercept=True, Cs=np.logspace(-4, 1, 50),
                                cv=3, penalty='l2', solver='lbfgs', tol=0.01))
])

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

result = model.get_params()['lr']
print('r：', model.score(x_train, y_train))
print('参数：', result.coef_)
print('截距：', result.intercept_)

# KNN模型训练
knn = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree', weights='distance')
knn.fit(x_train, y_train)
knn_y_predict = knn.predict(x_test)
print('knn-r：', knn.score(x_train, y_train))

t = np.arange(len(x_test))
plt.figure(figsize=(12, 9), facecolor='w')
plt.ylim(-0.5, 2.5)
plt.plot(t, y_test, 'ro', markersize=6, zorder=3, label=u'真实值')
plt.plot(t, y_predict, 'go', markersize=10, zorder=2, label=u'logistic预测值,$R^2$=%.3f' % result.score(x_test, y_test))
plt.plot(t, knn_y_predict, 'yo', markersize=16, zorder=1, label=u'knn预测值,$R^2$=%.3f' % knn.score(x_test, y_test))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'鸢尾花类别', fontsize=18)
plt.title(u'Logistic回归对比KNN预测鸢尾花类别', fontsize=20)
plt.show()