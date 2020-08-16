import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sample(num_of_samples, sample_sz):
    """从高斯分布中抽取num_of_samples个样本，每个样本容量sample_sz，返回num_of_samples样本卡方值"""
    data = []
    for _ in range(num_of_samples):
        data.append(np.sum(np.array([random.gauss(0.0, 1.0) for _ in range(sample_sz)])**2))
    return data


if __name__ == '__main__':

    data1 = sample(1000, 20)
    data2 = sample(1000, 50)
    data3 = sample(1000, 100)
    sns.distplot(data1, bins='auto')
    sns.distplot(data2, bins='auto')
    sns.distplot(data3, bins='auto')
    plt.legend(['n=20', 'n=50', 'n=100'])
    # plt.hist(data, rwidth=0.8, bins='auto')
    # plt.xlim(0,5000)
    plt.show()