# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 14:49
# @Author  : wangxg

from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)

RDD = sc.parallelize(['coffee', 'coffee', 'panda', 'monkey', 'tea'])
print(RDD.count())
print(RDD.countByValue())
print(RDD.countByKey())