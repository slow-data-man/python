# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 15:08
# @Author  : wangxg

from pyspark import SparkContext, SparkConf
import numpy as np

conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext(conf=conf)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def print_foreach(x):
    print(x)


numRDD = sc.parallelize(np.arange(-10, 10, 1))
numRDD1 = numRDD.repartition(5)
numRDD2 = numRDD1.map(sigmoid)
# foreach没有返回值，因此传入的函数不能进行计算操作
numRDD2.foreach(print_foreach)