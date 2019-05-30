# -*- coding: UTF-8 -*-# -*- coding: UTF-8 -*-
"""
    决策树预测波士顿房屋价格
    算法：决策树
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

from sklearn.externals.six import StringIO
from IPython.display import Image, display
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
ss = MinMaxScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

model = DecisionTreeRegressor(criterion='mae', max_depth=5)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print('score：', model.score(x_test, y_test))

# wget -c https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.msi
# pip install pydotplus
dot_data = tree.export_graphviz(model, out_file=None, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('dot.pdf')