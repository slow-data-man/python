# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 14:28
# @Author  : wangxg

from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)

numRDD = sc.parallelize([1, 3, 1, 4, 2, 0, 5])
numRDD1 = numRDD.repartition(3)
# take返回元素的顺序不能保证，且会得到不均衡集合
print(numRDD.take(6))
print(numRDD1.take(6))
# top返回前几个元素，默认降序
print(numRDD.top(6))
print(numRDD1.top(6))
# takeSample返回一个采样
print(numRDD.takeSample(False, 5, 666))
print(numRDD1.takeSample(False, 5, 6666))
