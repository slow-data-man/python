from playStats.descriptive_stats import mean
from  playStats.descriptive_stats import variance
import random
import matplotlib.pyplot as plt


def variance_bias(data):
    """有偏方差"""
    if data is None or len(data) <= 1:
        return None
    n = len(data)
    mean_value = mean(data)
    return sum((e - mean_value) ** 2 for e in data) / n


def sample(num_of_samples, sample_sz, var):
    """从均匀分布中抽取num_of_samples个样本，每个样本容量sample_sz，返回num_of_samples样本方差"""
    data = []
    for _ in range(num_of_samples):
        data.append(var([random.uniform(0.0,1.0) for _ in range(sample_sz)]))
    return data


if __name__ == '__main__':
    data1 = sample(1000, 40, variance_bias)
    data2 = sample(1000, 40, variance)

    plt.subplot(121)
    plt.hist(data1, bins="auto", rwidth=0.8)
    plt.axvline(x=mean(data1), c='y')
    plt.axvline(x=1/12, c='r')

    plt.subplot(122)
    plt.hist(data2, bins="auto", rwidth=0.8)
    plt.axvline(x=mean(data2), c='y')
    plt.axvline(x=1 / 12, c='r')
    plt.show()