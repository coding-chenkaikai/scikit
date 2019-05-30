# -*- coding: UTF-8 -*-
"""
    概率分布函数
"""

import numpy as np
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt

# 二项分布
def binomial_distribution():
    n = 10
    p = 0.3
    k = np.arange(0, 21)
    binomial = st.binom.pmf(k=k, n=n, p=p)

    plt.plot(k, binomial, 'o-')
    plt.title('Binomial: n=%i, p=%.2f' % (n, p))
    plt.show()

# 生成服从二项分布的随机变量
def binomial_distribution_rvs():
    data = st.binom.rvs(n=10, p=0.3, size=10000)
    plt.hist(data, 20, facecolor='g', alpha=0.75)
    plt.show()

# 泊松分布
def poisson_distribution():
    rate = 2
    n = np.arange(0, 11)
    poisson = st.poisson.pmf(n, rate)

    plt.plot(n, poisson, 'o-')
    plt.title('Poisson: rate=%i' % rate)
    plt.show()

# 生成服从泊松分布的随机变量
def poisson_distribution_rvs():
    data = st.poisson.rvs(mu=2, loc=0, size=10000)
    plt.hist(data, 20, facecolor='g', alpha=0.75)
    plt.show()

# 正态分布
def normal_distribution():
    mu = 0 # mean
    sigma = 1 # standard deviation
    x = np.arange(-5, 5, 0.1)
    normal = st.norm.pdf(x, mu, sigma)

    plt.plot(x, normal)
    plt.title('Normal: $\mu$=%.1f, $\sigma^2$=%.1f' % (mu, sigma))
    plt.grid(True)
    plt.show()

# beta分布
def beta_distribution():
    a = 0.5
    b = 0.5
    x = np.arange(0.01, 1, 0.01)
    beta = st.beta.pdf(x, a, b)

    plt.plot(x, beta)
    plt.title('Beta: a=%.1f, b=%.1f' % (a, b))
    plt.show()

# 指数分布
def exponential_distribution():
    l = 0.5
    x = np.arange(0, 15, 0.1)
    exponential = l * np.exp(-l * x)

    plt.plot(x, exponential)
    plt.title('Exponential: $\lambda$=%.2f' % l)
    plt.show()

# binomial_distribution()
# binomial_distribution_rvs()
# poisson_distribution()
# poisson_distribution_rvs()
# normal_distribution()
# beta_distribution()
exponential_distribution()