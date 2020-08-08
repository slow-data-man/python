# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 20:50
# @Author  : wangxg

from pyspark import SparkContext, SparkConf


conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
lines = sc.textFile("/usr/spark/README.md")
pythonLines = lines.filter(lambda line: "Python" in line)
for _ in pythonLines.collect():
    print(_)

# 打印 pythonLines 中的两个两个元素
for i in pythonLines.take(2):
    print(i)

# 打印 pythonLines 中元素个数
print(pythonLines.count())