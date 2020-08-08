# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 21:24
# @Author  : wangxg

from pyspark import SparkContext, SparkConf


conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
nums = sc.parallelize([1, 2, 3, 4])
square = nums.map(lambda x: x*x)
print(square.collect())

lines = sc.parallelize(['hello world', 'hi'])
words_map = lines.map(lambda line: line.split(" "))
words_flatMap = lines.flatMap(lambda line: line.split(" "))
print("words_map:", words_map.collect())
print("words_flatMap:", words_flatMap.collect())