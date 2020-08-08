# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 22:26
# @Author  : wangxg

from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext(conf=conf)
RDD1 = sc.parallelize([1, 1, 2, 2])
sumCount = RDD1.aggregate((4, 0),    # sum, cnt的初始值
                          (lambda acc, val: (acc[0] + val, acc[1] + 1)),    # acc=[sum, cnt],val=RDD1中的每一个元素，这个操作会使用一次初始值
                          (lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])))    # acc1与acc2为不同任务中的上一步产生的结果[sum,cnt]，这个操作也会使用一次初始值，因此初始值会被计算两次
sum, cnt = sumCount
avg = sum / cnt
print(avg)
print(RDD1.getNumPartitions())  # 查看RDD1的分区个数
