# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 20:38
# @Author  : wangxg

from pyspark import SparkContext, SparkConf


conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
inputRDD = sc.textFile("/var/spool/mail/root")
Node1RDD = inputRDD.filter(lambda line: "Node-1" in line)
Node2RDD = inputRDD.filter(lambda line: "Node-2" in line)
NodeLinesRDD = Node1RDD.union(Node2RDD)

print("inputRDD")
print(inputRDD.collect())

print("Node1RDD")
print(Node1RDD.collect())

print("Node2RDD")
print(Node2RDD.collect())

print("NodeLinesRDD")
print(NodeLinesRDD.collect())