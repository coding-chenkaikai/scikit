# -*- coding: UTF-8 -*-
"""
    Kmeans聚类
    算法：KMeans
    数据：
"""

import sklearn
import numpy as np

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_blobs

# 数据预处理
centers = 4
x, y = make_blobs(n_samples=1000, n_features=8, centers=centers, random_state=0)

# 模型训练
km = KMeans(n_clusters=centers, init='random', random_state=0)
km.fit(x)
print(km.score(x))