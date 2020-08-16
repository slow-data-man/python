"""一个正态总体的区间估计"""

from playStats.descriptive_stats import mean, variance, std
import numpy as np
from scipy.stats import norm, t, chi2, f


def mean_ci_est(data, alpha, sigma=None):
    """均值的区间估计"""
    n = len(data)
    sample_mean = mean(data)

    if sigma is None:
        # 方差未知
        s = std(data)
        me = s / np.sqrt(n)
        t_value = abs(t.ppf(alpha/2, n-1))
        return round(sample_mean - me * t_value, 2), round(sample_mean + me * t_value, 2)
    else:
        # 方差已知
        me = sigma / np.sqrt(n)
        z_value = abs(norm.ppf(alpha/2))
        return round(sample_mean - me * z_value, 2), round(sample_mean + me * z_value, 2)


def var_ci_est(data, alpha):
    """均值未知，方差的区间估计"""
    n = len(data)
    s2 = variance(data)
    chi2_low_value = chi2.ppf(alpha/2, n-1)
    chi2_high_value = chi2.ppf(1-alpha/2, n-1)
    return round((n-1)*s2/chi2_high_value,2), round((n-1)*s2/chi2_low_value,2)


"""两个正态总体的区间估计"""


def mean_diff_ci_t_est(data1, data2, alpha, equal=True):
    """总体方差未知， 求均值差的置信区间"""
    n1 = len(data1)
    n2 = len(data2)
    mean_diff = mean(data1) - mean(data2)
    sample1_var = variance(data1)
    sample2_var = variance(data2)

    if equal:
        """两总体方差未知且相等"""
        sw = np.sqrt(((n1-1)*sample1_var + (n2-1)*sample2_var) / (n1+n2-2))
        t_value = abs(t.ppf(alpha/2, n1+n2-2))
        return round(mean_diff - sw*np.sqrt(1/n1+1/n2)*t_value, 2), \
               round(mean_diff + sw*np.sqrt(1/n1+1/n2)*t_value, 2)
    else:
        """两总体方差未知且不等"""
        df = (sample1_var/n1 + sample2_var/n2)**2 / ((sample1_var/n1)**2 / (n1-1) + (sample2_var/n2)**2 / (n2-1))
        t_value = abs(t.ppf(alpha/2, df))
        return round(mean_diff - np.sqrt(sample1_var/n1 + sample2_var/n2) * t_value, 2), \
               round(mean_diff + np.sqrt(sample1_var/n1 + sample2_var/n2) * t_value, 2)


def mean_diff_ci_z_est(data1, data2, alpha, sigma1, sigma2):
    """两个总体方差已知，求均值差的置信区间"""
    n1 = len(data1)
    n2 = len(data2)
    mean_diff = mean(data1) - mean(data2)
    z_value = abs(norm.ppf(alpha/2))
    return round(mean_diff - np.sqrt(sigma1**2/n1 + sigma2**2/n2) * z_value, 2), \
           round(mean_diff + np.sqrt(sigma1**2/n1 + sigma2**2/n2) * z_value, 2)


def var_ratio_ci_est(data1, data2, alpha):
    """两个总体方差未知，求方差比的置信区间"""
    n1 = len(data1)
    n2 = len(data2)
    sample_ratio = variance(data1) / variance(data2)
    f_low_value = f.ppf(alpha/2, n1-1, n2-1)
    f_high_value = f.ppf(1-alpha/2, n1-1, n2-1)
    return round(sample_ratio / f_high_value, 3), round(sample_ratio / f_low_value, 3)
