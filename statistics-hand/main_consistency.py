import random
from playStats.descriptive_stats import mean
from playStats.descriptive_stats import variance
import matplotlib.pyplot as plt


if __name__ == '__main__':

    indices = []
    data_mean = []
    data_varvance = []
    for sample_sz in range(20, 10001, 50):
        indices.append(sample_sz)
        sample = [random.gauss(0.0, 1.0) for _ in range(sample_sz)]
        data_mean.append(mean(sample))
        data_varvance.append(variance(sample))

    plt.plot(indices, data_mean)
    plt.axhline(0, c='r')

    plt.plot(indices, data_varvance)
    plt.axhline(1, c='b')

    plt.show()