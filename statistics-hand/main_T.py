import random
import numpy as np
import matplotlib.pyplot as plt
from playStats.descriptive_stats import mean, variance


def sample(num_of_samples, sample_sz):
    """从高斯分布中抽取num_of_samples个样本，每个样本容量sample_sz，返回num_of_samples样本卡方值"""
    if sample_sz < 1:
        return None
    data_X = []
    data_Y = []
    for _ in range(num_of_samples):
        # samples = [random.gauss(0.0,1.0) for _ in range(sample_sz)]
        # data_X.append(mean(samples) / (1/np.sqrt(sample_sz)))
        # data_Y.append(variance(samples)*(sample_sz-1)/1.0)
        data_X.append(random.gauss(0.0,1.0))
        data_Y.append(sum([e**2 for e in [random.gauss(0.0,1.0) for _ in range(sample_sz)]]))
    return np.array(data_X), np.array(data_Y)


def f(x):
    """标准正态分布函数"""
    return np.exp(-x**2/2)/np.sqrt(2*np.pi)


if __name__ == '__main__':
    num_of_samples, sample_sz = 1000, 10
    X, Y = sample(num_of_samples, sample_sz)
    t = X / np.sqrt(Y/(sample_sz))
    print(max(t), min(t))

    plt.subplot(311)
    plt.hist(t, rwidth=0.8, bins='auto', density=True)
    plt.plot(np.arange(-4, 4, 0.01), f(np.arange(-4, 4, 0.01)))

    num_of_samples, sample_sz = 1000, 30
    X, Y = sample(num_of_samples, sample_sz)
    t = X / np.sqrt(Y / (sample_sz))
    print(max(t), min(t))

    plt.subplot(312)
    plt.hist(t, rwidth=0.8, bins='auto', density=True)
    plt.plot(np.arange(-4, 4, 0.01), f(np.arange(-4, 4, 0.01)))

    num_of_samples, sample_sz = 1000, 70
    X, Y = sample(num_of_samples, sample_sz)
    t = X / np.sqrt(Y / (sample_sz))
    print(max(t), min(t))

    plt.subplot(313)
    plt.hist(t, rwidth=0.8, bins='auto', density=True)
    plt.plot(np.arange(-4, 4, 0.01), f(np.arange(-4, 4, 0.01)))

    plt.show()