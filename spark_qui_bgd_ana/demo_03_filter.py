# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 20:31
# @Author  : wangxg


from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
inputRDD = sc.textFile("/var/spool/mail/root")
errorRDD = inputRDD.filter(lambda line: "WARN" in line)
sout = errorRDD.collect()
for i in sout:
    print(i)
