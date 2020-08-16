"""验证一个正态总体均值与方差的区间估计"""


from playStats.descriptive_stats import mean, variance, std
from playStats.interval_est import mean_ci_est, var_ci_est, mean_diff_ci_t_est, mean_diff_ci_z_est, var_ratio_ci_est


if __name__ == '__main__':

    salary_18 = [1484, 785, 1598, 1366, 1716, 1020, 1716, 785, 3113, 1601]
    salary_35 = [902, 4508, 3809, 3923, 4276, 2065, 1601, 553, 3345, 2182]

    # print("salary_18均值的点估计：", mean(salary_18), "；置信区间：", mean_ci_est(salary_18, 0.05))
    # print("salary_35均值的点估计：", mean(salary_35), "；置信区间：", mean_ci_est(salary_35, 0.05))
    #
    # print("salary_18标准差、方差的点估计：", round(std(salary_18),2), round(variance(salary_18),2), \
    #       "；置信区间：", var_ci_est(salary_18, 0.05))
    # print("salary_35标准差、方差的点估计：", round(std(salary_35),2), round(variance(salary_35),2), \
    #       "；置信区间：", var_ci_est(salary_35, 0.05))

    # print(mean(salary_18)-mean(salary_35))
    # print(mean_diff_ci_t_est(salary_18, salary_35, 0.05, equal=True))
    # print(mean_diff_ci_t_est(salary_18, salary_35, 0.05, equal=False))

    print(mean_diff_ci_z_est(salary_18, salary_35, 0.05, 1035, 1240))

    print(var_ratio_ci_est(salary_18, salary_35, 0.05))