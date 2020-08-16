import matplotlib.pyplot as plt
import random
import numpy as np


def toss():
    """模拟抛硬币"""
    return random.randint(0,1)  # 随机产生0与1，0表示硬币正面


if __name__ == '__main__':

    indices = []
    freq = []
    for toss_num in range(10,10001,10):
        heads = 0
        for _ in range(toss_num):
            if toss() == 0:
                heads += 1
        freq.append(heads/toss_num)  # 计算正面朝上的频率
        indices.append(toss_num)
    plt.plot(indices, freq)
    plt.plot(indices,0.5*np.ones(len(indices)), 'r')
    plt.show()