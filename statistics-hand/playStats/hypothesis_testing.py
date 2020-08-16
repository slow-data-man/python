from playStats.descriptive_stats import mean, std, variance, cor
import numpy as np
from scipy.stats import norm, t, chi2, f


def z_test(data1, data2=None, tail="both", mu=0.0, sigma1=1.0, sigma2=None):

    assert tail in ['both', 'left', 'right'], 'tail should be one of "both", "left", "right"'

    if data2 is None:
        # 单个总体的情况
        mean_val = mean(data1)
        se = sigma1 / np.sqrt(len(data1))
        z_val = (mean_val - mu) / se
    else:
        # 两个总体的情况
        assert sigma2 is not None
        mean_diff = mean(data1) - mean(data2)
        se = np.sqrt(sigma1**2 / len(data1) + sigma2**2 / len(data2))
        z_val = (mean_diff - mu) / se

    if tail == 'both':
        # 双尾检验
        p = 2 * (1 - norm.cdf(abs(z_val)))
    elif tail == 'left':
        # 左尾检验
        p = norm.cdf(z_val)
    else:
        # 右尾检验
        p = 1 - norm.cdf(z_val)

    return round(z_val, 2), p


def t_test(data1, data2=None, tail='both', mu=0.0, equal=True):

    assert tail in ['both', 'left', 'right'], 'tail should be one of "both", "left", "right"'

    if data2 is None:
        # 单个总体的情况
        mean_val = mean(data1)
        se = std(data1) / np.sqrt(len(data1))
        t_val = (mean_val - mu) / se
        df = len(data1) - 1
    else:
        # 两个总体的情况
        n1 = len(data1)
        n2 = len(data2)
        mean_diff = mean(data1) - mean(data2)
        sample1_var = variance(data1)
        sample2_var = variance(data2)
        if equal:
            # 方差相等的情况
            sw = np.sqrt((((n1-1)*sample1_var + (n2-1)*sample2_var)) / (n1+n2-2))
            t_val = (mean_diff - mu) / (sw * np.sqrt(1/n2 + 1/n2))
            df = n1 + n2 -2
        else:
            # 方差不等的情况
            se = np.sqrt(sample1_var/n1 + sample2_var/n2)
            t_val = (mean_diff - mu) / se
            df = (sample1_var / n1 + sample2_var / n2) ** 2 / (
                        (sample1_var / n1) ** 2 / (n1 - 1) + (sample2_var / n2) ** 2 / (n2 - 1))

    if tail == 'both':
        # 双尾检验
        p = 2 * (1 - t.cdf(abs(t_val), df))
    elif tail == 'left':
        # 左尾检验
        p = t.cdf(t_val, df)
    else:
        # 右尾检验
        p = 1 - t.cdf(t_val, df)

    return round(t_val, 2), round(df, 2), p


def t_test_paired(data1, data2, tail='both', mu=0):
    # 成对数据的检验
    data = [e1-e2 for e1,e2 in zip(data1, data2)]
    return t_test(data, tail=tail, mu=mu)


def chi2_test(data, tail='both', sigma2=1):
    """单个总体"""
    assert tail in ['both', 'left', 'right'], 'tail should be one of "both", "left", "right"'

    n = len(data)
    sample_var = variance(data)
    chi2_val = (n-1) * sample_var / sigma2

    if tail == 'both':
        # 双尾检验
        p = 2 * min(1 - chi2.cdf(chi2_val, n-1), chi2.cdf(chi2_val, n-1))
    elif tail == 'left':
        # 左尾检验
        p = chi2.cdf(chi2_val, n-1)
    else:
        # 右尾检验
        p = 1 - chi2.cdf(chi2_val, n-1)

    return round(chi2_val, 2), n-1, p


def f_test(data1, data2, tail='both', ratio=1):
    """两个总体"""
    assert tail in ['both', 'left', 'right'], 'tail should be one of "both", "left", "right"'
    n1 = len(data1)
    n2 = len(data2)
    sample_var1 = variance(data1)
    sample_var2 = variance(data2)
    f_val = sample_var1 / sample_var2 / ratio

    if tail == 'both':
        # 双尾检验
        p = 2 * min(1- f.cdf(f_val, n1-1, n2-1), f.cdf(f_val, n1-1, n2-1))
    elif tail == 'left':
        # 左尾检验
        p = f.cdf(f_val, n1-1, n2-1)
    else:
        # 右尾检验
        p = 1 - f.cdf(f_val, n1-1, n2-1)

    return round(f_val,4), n1-1, n2-1, round(p,5)


def anova_oneway(data):
    """单因素方差分析"""
    k = len(data)  # 类别数
    assert k>1, '数据量得大于1'

    group_means = [mean(group) for group in data]
    group_szs = [len(group) for group in data]
    n = sum(group_szs)  # 每个类别中元素个数之和，即数据总个数
    assert n>k

    group_mean = sum(group_mean * group_sz for group_mean, group_sz in zip(group_means, group_szs)) / n
    sst = np.sum((np.array(data) - group_mean)**2)
    ssg = ((np.array(group_means) - group_mean)**2).dot(np.array(group_szs))
    sse = np.sum((np.array(data) - np.array(group_means).reshape(-1,1))**2)
    assert round(sse, 2) == round(sst - ssg, 2)

    dfg = k-1
    dfe = n-k

    msg = ssg/dfg
    mse = sse/dfe

    f_value = msg/mse
    p = 1 - f.cdf(f_value, dfg, dfe)

    return round(f_value, 2), dfg, dfe, p


def anova_twoway(data):
    """双因素方差分析2×2"""
    r, s = 2, 2
    data = np.array(data)
    group_szs = np.tile(np.size(data, axis=1),(np.size(data, axis=0), 1))
    n = sum(group_szs)  # 样本总数

    # 计算均值
    group_means = np.mean(data, axis=1)
    group_mean = group_means.dot(group_szs) / n
    group_i_means = np.array([mean(group_means[:2]), mean(group_means[2:])])
    group_j_means = np.array([(group_means[0]+group_means[2])/2, (group_means[1]+group_means[3])/2])

    # 计算i，j各水平的效应
    group_i_effect = group_i_means - group_mean
    group_j_effect = group_j_means - group_mean
    # 计算i, j的交叉效应
    group_ij_effect = (group_means.reshape(2,2) - np.tile(group_mean, (2, 2))) - np.tile(group_i_effect, (2, 1)).T - np.tile(group_j_effect, (2, 1))

    # 计算总变化
    sst = np.sum((data - group_mean)**2)
    # 计算第一个因素引起的变化
    ss_method = ((group_i_means - group_mean)**2).dot([np.sum(group_szs[:2]), np.sum(group_szs[2:])])
    # 计算第二个因素引起的变化
    ss_reward = ((group_j_means - group_mean)**2).dot([np.sum([group_szs[0], group_szs[2]]), np.sum([group_szs[1], group_szs[3]])])
    # 计算第一个因素与第二个因素交互引起的变化
    ss_mr = (group_ij_effect.reshape(1,4)**2).dot(group_szs)
    # 其他因素引起的变化
    ss_error = np.sum((data - group_means.reshape(-1, 1))**2)

    # 计算其他因素引起的误差
    ms_error = ss_error / (n-r*s)
    # 计算第一个因素引起的变化ms值, f值, p值
    ms_method = ss_method / (r-1)
    f_ms_method = ms_method / ms_error
    p_ms_method = 1 - f.cdf(f_ms_method, r-1, n-r*s)
    # 计算第二个因素引起的变化ms值, f值, p值
    ms_reward = ss_reward / (r - 1)
    f_ms_reward = ms_reward / ms_error
    p_ms_reward = 1 - f.cdf(f_ms_reward, r - 1, n - r * s)
    # 计算第一、二个因素交互引起的变化ms值, f值, p值
    ms_mr = ss_mr / (r - 1)
    f_ms_mr = ms_mr / ms_error
    p_ms_mr = 1 - f.cdf(f_ms_mr, r - 1, n - r * s)

    # 整理输出矩阵各行
    method = [r-1, ss_method, ms_method, f_ms_method, p_ms_method]
    reward = [r-1, ss_reward, ms_reward, f_ms_reward, p_ms_reward]
    mr = [r-1, ss_mr, ms_mr, f_ms_mr, p_ms_mr]
    residuals = [n-r*s, ss_error, ms_error, None, None]

    return np.array([method, reward, mr, residuals]).astype(np.float32)


def cor_test(data1, data2):
    """相关系数的假设检验"""
    r = cor(data1, data2)
    n = len(data1)
    t_val = r / (np.sqrt(1-r**2) / np.sqrt(n-2))
    p = 2 * (1 - abs(t.cdf(t_val, n-2)))

    return round(r, 3), round(t_val, 3), n-2, p