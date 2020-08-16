from playStats.hypothesis_testing import z_test, t_test, t_test_paired, chi2_test, f_test
from playStats.hypothesis_testing import anova_oneway, anova_twoway, cor_test
from playStats.interval_est import mean_ci_est, mean_diff_ci_z_est


if __name__ == '__main__':

    data1 = [41, 36, 12, 18, 23, 19, 8, 16, 11, 14, 18, 14, 34, 6, 30, 11, 1, 11, 4, 32]
    data2 = [23, 45, 115, 37, 29, 71, 39, 23, 21, 37, 20, 12, 13, 135, 49, 32, 64, 40, 77, 97]

    # one-sample z-test
    # print(z_test(data1, tail='both', mu=35, sigma1=5))
    # print(mean_ci_est(data1, 0.05, sigma=5))

    # two-sample z-test
    # print(z_test(data1, data2, tail='both', mu=0.0, sigma1=5, sigma2=15))
    # print(mean_diff_ci_z_est(data1, data2, 0.05, sigma1=5, sigma2=15))

    # one-sample t-test
    # print(t_test(data1, tail='both', mu=35))

    # two-sample t-test
    # print(t_test(data1, data2, tail='both', mu=0.0, equal=True))
    # print(t_test(data1, data2, tail='both', mu=0.0, equal=False))

    # pair-sample t-test
    # print(t_test_paired(data1, data2, tail='both', mu=0.0))

    # one-sample chi2-test
    # print(chi2_test(data1, tail='both', sigma2=5))

    # two variance f-test
    # print(f_test(data1, data2, tail='both', ratio=1))

    # anova_oneway analysis
    # data = [
    #     [77, 88, 77, 85, 81, 72, 80, 80, 76, 84],
    #     [74, 88, 77, 93, 91, 95, 85, 88, 93, 79],
    #     [93, 94, 95, 83, 94, 94, 85, 91, 90, 96]
    # ]
    # print(anova_oneway(data))

    # anova_oneway vs t_test
    # data1 = [77, 88, 77, 85, 81, 72, 80, 80, 76, 84]
    # data2 = [74, 88, 77, 93, 91, 95, 85, 88, 93, 79]
    # print(t_test(data1, data2, tail='both', equal=True, mu=0))
    # print(anova_oneway([data1, data2]))
    # 这个对比表明：t_test 是单因素方差分析(anova_oneway)的特殊情况

    # anova_twoway analysis
    # data = [
    #     [77, 88, 77, 85, 81, 72, 80, 80, 76, 84],
    #     [96, 87, 94, 90, 80, 99, 100, 87, 96, 95],
    #     [93, 94, 95, 83, 94, 94, 85, 91, 90, 96],
    #     [74, 88, 77, 93, 91, 95, 85, 88, 93, 79]
    # ]
    # print(anova_twoway(data))

    # cor t-test
    score = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    happy = [1, 3, 2, 6, 4, 5, 8, 10, 9, 7]
    print(cor_test(score, happy))