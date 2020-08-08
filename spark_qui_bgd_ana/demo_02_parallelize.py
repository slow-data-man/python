# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 20:27
# @Author  : wangxg


from pyspark import SparkContext, SparkConf


conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
lines = sc.parallelize(["pandas", "i like pandas"])     # 创建RDD
sout = lines.collect()
print(sout)