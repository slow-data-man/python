# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 22:04
# @Author  : wangxg

from pyspark import SparkContext, SparkConf


conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)

RDD1 = sc.parallelize(['coffee', 'coffee', 'panda', 'monkey', 'tea'])
RDD2 = sc.parallelize(['coffee', 'monkey', 'kitty'])

print('distinct:', RDD1.distinct().collect())
print('union:', RDD1.union(RDD2).collect())
print('intersection:', RDD1.intersection(RDD2).collect())
print('subtract:', RDD1.subtract(RDD2).collect())
print('cartesian:', RDD1.cartesian(RDD2).collect())