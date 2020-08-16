from collections import Counter
from math import sqrt


def frequncy(data):
    """频率"""
    counter = Counter(data)
    ret = []
    for point in counter.most_common():
        ret.append((point[0], point[1]/len(data)))
    return ret


def mode(data):
    """众数"""
    counter = Counter(data)
    if counter.most_common()[0][1] == 1:
        print("mode is not available")
        return None,None

    count = counter.most_common()[0][1]    # 众数出现的次数
    ret = []
    for point in counter.most_common():
        if point[1] == count:
            ret.append(point[0])
        else:
            break
    return ret, count


def median(data):
    """中位数"""
    sorted_data = sorted(data)
    n = len(sorted_data)

    if n % 2 == 1:
        return sorted_data[n//2]

    return (sorted_data[n//2-1] + sorted_data[n//2]) / 2


def mean(data):
    """均值"""
    if data is None:
        return None
    return sum(data) / len(data)


def rng(data):
    """极差"""
    if data is None:
        return None
    return max(data) - min(data)


def quartile(data):
    if data is None or len(data) < 4:
        return None
    q1, q2, q3 = None, None, None
    n = len(data)
    sorted_data = sorted(data)
    q2 = median(sorted_data)
    if n % 2 == 1:
        q1 = median(sorted_data[:n//2])
        q3 = median(sorted_data[n//2+1:])
    else:
        q1 = median(sorted_data[:n//2])
        q3 = median(sorted_data[n//2:])
    return q1, q2, q3


def variance(data):
    """方差"""
    if data is None or len(data) <= 1:
        return None
    n = len(data)
    mean_value = mean(data)
    return sum((e - mean_value) ** 2 for e in data) / (n-1)


def std(data):
    """标准差"""
    return sqrt(variance(data))


def covariance(data1, data2):
    """协方差"""
    if data1 is None or data2 is None:
        return None
    n1 = len(data1)
    n2 = len(data2)
    assert n1 == n2

    data1_mean = mean(data1)
    data2_mean = mean(data2)
    return round(sum([(e1-data1_mean)*(e2-data2_mean) for e1, e2 in zip(data1, data2)]) / (n1-1), 3)


def cor(data1, data2):
    """相关系数"""
    if data1 is None or data2 is None:
        return None
    n1 = len(data1)
    n2 = len(data2)
    assert n1 == n2

    std1 = std(data1)
    std2 = std(data2)

    return round(covariance(data1, data2) / (std1 * std2), 3)