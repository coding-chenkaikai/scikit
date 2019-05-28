# -*- coding: UTF-8 -*-
"""
    决策树预测鸢尾花类别
    算法：决策树
    数据：datas/iris.data
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt
import sklearn.preprocessing as pre_processing

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model.coordinate_descent import ConvergenceWarning

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

# 模型训练
model = Pipeline([
    ('ss', MinMaxScaler()),
    ('skb', SelectKBest(chi2)),
    ('pca', PCA()),
    ('decision', DecisionTreeClassifier(random_state=0))
])

# parameters = {
#     'skb__k': [1, 2, 3, 4],
#     'pca__n_components': [0.5, 0.99],
#     'decision__criterion': ['gini', 'entropy'],
#     'decision__max_depth': np.arange(1, 11)
# }
#
# gscv = GridSearchCV(model, param_grid=parameters, cv=3)
# gscv.fit(x_train, y_train)
# print('最优参数：', gscv.best_params_)
# print('score：', gscv.best_score_)
# print('最优模型：', gscv.best_estimator_)

model.set_params(skb__k=4, pca__n_components=0.5, decision__criterion='gini', decision__max_depth=2)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print('score：', model.score(x_train, y_train))

t = np.arange(len(x_test))
plt.figure(figsize=(14, 7), facecolor='w')
plt.ylim(-0.5, 2.5)
plt.plot(t, y_test, 'ro', markersize=6, zorder=3, label=u'真实值')
plt.plot(t, y_predict, 'go', markersize=10, zorder=2, label=u'预测值,$R^2$=%.3f' % model.score(x_test, y_test))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'鸢尾花类别', fontsize=18)
plt.title(u'决策树预测鸢尾花类别', fontsize=20)
plt.show()