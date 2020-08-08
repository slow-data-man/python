# -*- coding: utf-8 -*-
# @Time    : 2020/5/29 20:26
# @Author  : wangxg
# @Email   : 
# @File    : demo_01_textFile.py
# @Software: PyCharm


from pyspark import SparkContext, SparkConf


conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
lines = sc.textFile("/usr/spark/README.md")     # 创建RDD
pythonLines = lines.filter(lambda line:"Python" in line)
print(pythonLines.first())