"""验证中心极限定理"""
import random
import matplotlib.pyplot as plt
from  playStats.descriptive_stats import mean


def sample(num_of_samples, sample_sz):
    """从均匀分布中抽取num_of_samples个样本，每个样本容量sample_sz，返回num_of_samples样本均值"""
    data = []
    for _ in range(num_of_samples):
        data.append(mean([random.uniform(0.0,1.0) for _ in range(sample_sz)]))
    return data


if __name__ == '__main__':
    data = sample(2000, 40)
    plt.hist(data, bins='auto', rwidth=0.8)
    plt.axvline(x=mean(data), c='red')  # 画出样本均值线
    plt.show()