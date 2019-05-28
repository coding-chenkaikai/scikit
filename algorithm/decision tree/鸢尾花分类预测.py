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