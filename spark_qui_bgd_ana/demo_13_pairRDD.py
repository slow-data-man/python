# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 15:45
# @Author  : wangxg

from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
lines = sc.textFile("/usr/spark/README.md")

pairRDD = lines.map(lambda x: (x.split(" ")[0], x))
# 通过foreach打印pairRDD中的元素
pairRDD.foreach(lambda x: print(x) if x[0] != '' else None)