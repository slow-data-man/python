# -*- coding: utf-8 -*-
# @Time    : 2020/5/31 11:39
# @Author  : wangxg

from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
lines = sc.textFile("/usr/spark/README.md")

# 通过 partitionBy()、spark.HashPartitioner 将大表转换成哈希分区
sc.sequenceFile().partitionBy(spark.HashPartitioner(100)).cache()