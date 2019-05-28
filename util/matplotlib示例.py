# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# 平方函数曲线
def square():
    x = np.linspace(-5, 5, 100)
    # y = x ** 2
    y = np.square(x)
    plt.plot(x, y, 'r-', linewidth=2, label='square curve')

    # y = 1 / (1 + np.exp(-x))
    # plt.plot(x, y, 'r-', linewidth=2, label='sigmoid curve')

    # y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    # plt.plot(x, y, 'r-', linewidth=2, label='tanh curve')

    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('x**2')
    plt.show()

# 模拟数据
def generate_data():
    x = np.random.uniform(-3, 3, size=100)
    X = x.reshape(-1, 1)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    plt.scatter(x, y)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# 心形
def heart():
    t = np.linspace(0, 2 * np.pi, 100)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    plt.plot(x, y, 'r-', linewidth=2)
    plt.grid(True)
    plt.show()

# 直方图
def histogram():
    # u = np.random.uniform(0.0, 1.0, 10000)
    u = np.random.randn(10000)
    plt.hist(u, 80, facecolor='g', alpha=0.75)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # square()
    generate_data()
    # heart()
    # histogram()
