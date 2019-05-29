# -*- coding: UTF-8 -*-
"""
    随机森林预测宫颈癌
    算法：随机森林
    数据：data/risk_factors_cervical_cancer.csv
"""

import time
import warnings
import pydotplus
import numpy as np
import pandas as pd
import matplotlib as mpl

import matplotlib.pyplot as plt
import sklearn.preprocessing as pre_processing
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV

# 防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 拦截异常
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# 数据预处理
path = "datas/risk_factors_cervical_cancer.csv"
names = [u'Age', u'Number of sexual partners', u'First sexual intercourse',
         u'Num of pregnancies', u'Smokes', u'Smokes (years)',
         u'Smokes (packs/year)', u'Hormonal Contraceptives',
         u'Hormonal Contraceptives (years)', u'IUD', u'IUD (years)', u'STDs',
         u'STDs (number)', u'STDs:condylomatosis',
         u'STDs:cervical condylomatosis', u'STDs:vaginal condylomatosis',
         u'STDs:vulvo-perineal condylomatosis', u'STDs:syphilis',
         u'STDs:pelvic inflammatory disease', u'STDs:genital herpes',
         u'STDs:molluscum contagiosum', u'STDs:AIDS', u'STDs:HIV',
         u'STDs:Hepatitis B', u'STDs:HPV', u'STDs: Number of diagnosis',
         u'STDs: Time since first diagnosis', u'STDs: Time since last diagnosis',
         u'Dx:Cancer', u'Dx:CIN', u'Dx:HPV', u'Dx', u'Hinselmann', u'Schiller',
         u'Citology', u'Biopsy']
df = pd.read_csv(path)
df = df.replace('?', np.NAN)

x = df[names[0:-4]]
y = df[names[-4:]]
x = pre_processing.Imputer(missing_values=np.NAN).fit_transform(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = Pipeline([
    ('ss', MinMaxScaler()),
    ('pca', PCA()),
    ('forest', RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=1, random_state=0))
])

model.set_params(pca__n_components=2)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print('score：', model.score(x_test, y_test))
