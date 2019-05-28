# -*- coding: UTF-8 -*-

import numpy as np

# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid函数导数
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# tanh函数
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# tanh函数导数
def tanh_derivative(x):
    return 1 - tanh(x) * tanh(x)

# relu函数
def relu(x):
    return np.where(x < 0, 0, x)

# relu函数导数
def relu_derivative(x):
    return np.where(x < 0, 0, 1)

# 规范化
def normalize_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

# softmax
def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)
