# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 22:21
# @Author  : wangxg

from pyspark import SparkContext, SparkConf


conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
numRDD = sc.parallelize([1, 2, 3, 4, 5, 6])
res = numRDD.reduce(lambda x, y: x+y)
print(res)