# -*- coding: UTF-8 -*-
"""
    决策树预测波士顿房屋价格
    算法：决策树
    数据：data/boston_housing.data
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt
import sklearn.preprocessing as pre_processing

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
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

# 拦截异常
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# 数据预处理
path = "datas/boston_housing.data"
names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
df = pd.read_csv(path, header=None, names=names)
df = df.replace('?', np.NAN).dropna(axis=0, how='any')

x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 模型训练
models = [
    Pipeline([
        ('ss', MinMaxScaler()),
        ('pca', PCA()),
        ('decision', DecisionTreeRegressor(random_state=0))
    ]),
    Pipeline([
        ('ss', MinMaxScaler()),
        ('decision', DecisionTreeRegressor(random_state=0))
    ])
]

# parameters = [
#     {
#         'pca__n_components': [0.25, 0.5, 0.75, 1],
#         'decision__criterion': ['mse', 'mae'],
#         'decision__max_depth': np.arange(1, 11)
#     },
#     {
#         'decision__criterion': ['mse', 'mae'],
#         'decision__max_depth': np.arange(1, 11)
#     }
# ]
#
# for i in np.arange(len(models)):
#     gscv = GridSearchCV(models[i], param_grid=parameters[i])
#     gscv.fit(x_train, y_train)
#     print('最优参数：', gscv.best_params_)
#     print('score：', gscv.best_score_)
#     print('最优模型：', gscv.best_estimator_)

model = models[1]
model.set_params(decision__criterion='mae', decision__max_depth=3)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print('score：', model.score(x_train, y_train))

