from collections import Counter
from playStats.descriptive_stats import frequncy
from playStats.descriptive_stats import mode
from playStats.descriptive_stats import median
from playStats.descriptive_stats import mean
from playStats.descriptive_stats import rng
from playStats.descriptive_stats import quartile
from playStats.descriptive_stats import variance, std
from playStats.descriptive_stats import covariance, cor


if __name__ == '__main__':

    data = [2, 2, 2, 2, 1, 1, 1, 3, 3]
    # # 测试频数
    # count = Counter(data)
    # print(count.most_common())
    # print("data的众数为：", count.most_common()[0][0])
    #
    # # 测试频率
    # freq = frequncy(data)
    # print(freq)

    # 测试众数
    # mode_elements = mode(data)
    # print(mode_elements)

    # 测试中位数
    # data_1 = [1,4,2,3]
    # data_2 = [1,6,3,4,5]
    # data_3 = [1,6,3,4,5, 99]
    # print(median(data_1))
    # print(median(data_2))
    # print(median(data_3))

    # 测试均值
    # data_1 = [1, 6, 3, 4, 5]
    # data_2 = [1, 6, 3, 4, 5, 99]
    # print(mean(data_1))
    # print(mean(data_2))    # 均值受极端值影响很大

    # 测试极差
    # data_1 = [1, 6, 3, 4, 5]
    # data_2 = [1, 6, 3, 4, 5, 99]
    # print(rng(data_1))
    # print(rng(data_2))    # 极差受极端值影响很大

    # 测试分位数
    # data_1 = [1,2,3,4,5,6]
    # data_2 = [1,2,3,4,5,6,7]
    # print(quartile(data_1))
    # print(quartile(data_2))

    # 测试方差, 标准差
    # data = [1, 2, 3, 4, 5, 6, 7]
    # var = variance(data)
    # print(var)
    # print(std(data))

    # 测试协方差，相关系数
    score = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    happy = [1, 3, 2, 6, 4, 5, 8, 10, 9, 7]
    print(covariance(score, happy))
    print(cor(score, happy))
