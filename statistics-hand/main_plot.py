import matplotlib.pyplot as plt
import random
from collections import Counter


if __name__ == '__main__':

    random.seed(666)

    # scatter plot 散点图
    # x = [random.randint(0,100) for i in range(100)]
    # y = [random.randint(0,100) for i in range(100)]
    # plt.scatter(x, y)
    # plt.show()

    # line plot 线型图
    # x = [random.randint(0, 100) for i in range(100)]
    # plt.plot([i for i in range(100)],x)
    # plt.show()

    # bar plot 条形图：分类变量
    # data = [3,3,4,1,5,4,2,1,5,4,4,4,5,3,2,1,4,5,5]
    # counter = Counter(data)
    # x = [point[0] for point in counter.most_common()]
    # y = [point[1] for point in counter.most_common()]
    # plt.bar(x, y)
    # plt.show()

    # histogram 频率直方图：数值变量
    # data = [random.randint(0,100) for i in range(1000)]
    # plt.hist(data, rwidth=0.8, bins=5, density=True)
    # plt.show()

    # box plot 修正箱线图
    # data = [random.randint(0, 100) for i in range(1000)]
    # data.append(200)
    # data.append(-200)
    # plt.boxplot(data)
    # plt.show()

    # side-by-side box plot 并排箱线图
    data_1 = [random.randint(66, 166) for i in range(200)]
    data_2 = [random.randint(60, 120) for i in range(200)]
    plt.boxplot([data_1, data_2])
    plt.show()