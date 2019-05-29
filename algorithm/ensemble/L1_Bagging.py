# -*- coding: UTF-8 -*-
"""
    Bagging预测波士顿房屋价格
    算法：Bagging
    数据：data/boston_housing.data
"""

import time
import warnings
import pydotplus
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

from IPython.display import Image, display
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor

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

model = Pipeline([
    # ('lr', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=True))

    # ('bg', BaggingRegressor(RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=True), max_features=0.8,
    #                         max_samples=0.7, n_estimators=50, random_state=0))

    # ('adr', AdaBoostRegressor(LinearRegression(), n_estimators=100, learning_rate=0.0001, random_state=0))

    ('gbdt', GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, random_state=0))
])

model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print('train score：', model.score(x_train, y_train))
print('test score：', model.score(x_test, y_test))