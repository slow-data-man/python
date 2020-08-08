# -*- coding: utf-8 -*-
# @Time    : 2020/5/30 15:52
# @Author  : wangxg
from typing import Any, Union

from pyspark import SparkContext, SparkConf, RDD
import numpy as np

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
lines = sc.textFile("/usr/spark/README.md")
pairRDD = lines.map(lambda x: (x.split(" ")[0], x))

# reduceByKey，合并具有相同键的值
# pairRDD1 = sc.parallelize([('a', 3), ('b', 4), ('a', 1.5), ('c', 6), ('a', 2.5)])
# pairRDD1_res = pairRDD1.reduceByKey(lambda x, y: x+y)    # 通过函数定义合并方法
# print(pairRDD1_res.collect())

# groupByKey，对具有相同键的值进行分组
# pairRDD2 = sc.parallelize([('a', 3), ('b', 4), ('a', 1.5), ('c', 6), ('a', 2.5)])
# pairRDD2_res = pairRDD2.groupByKey()    # 通过函数定义合并方法
# pairRDD2_res.foreach(lambda x: print(list(x[1])))    # 打印结果

# 计算每个键的平均值，reduceByKey, mapValues
# pairRDD3 = sc.parallelize([('a', 3), ('b', 4), ('a', 1.5), ('c', 6), ('a', 2.5)])
# tmp1 = pairRDD3.mapValues(lambda x: (x, 1))
# tmp2 = tmp1.reduceByKey(lambda x1, x2: (x1[0] + x2[0], x1[1] + x2[1]))
# tmp3 = tmp2.mapValues(lambda x: 1.0 * x[0] / x[1])
# print(tmp3.collect())
# # 将上述三步写在一起
# pairRDD3_res = pairRDD3.mapValues(lambda x: (x, 1)) \
#                        .reduceByKey(lambda x1, x2: (x1[0] + x2[0], x1[1] + x2[1])) \
#                        .mapValues(lambda x: 1.0 * x[0] / x[1])
# print(pairRDD3_res.collect())

# 单词计数 flatMap, countByValue
# words = lines.flatMap(lambda x: x.split(" "))
# res = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y, 10)    # 10为并行度参数
# # print(res.getNumPartitions())
# # res.foreach(lambda x: print(x))
# # 使用 countByValue 进行单词计数
# res = words.flatMap(lambda x: x.split(" ")).countByValue()    # 返回的是 defaultdict 类型
# for i in res:
#     print(i, res[i])

# 计算每个键对应的平均值 combineByKey
# pairRDD3 = sc.parallelize([('a', 3), ('b', 4), ('a', 1.5), ('c', 6), ('a', 2.5)])
# sumCount = pairRDD3.combineByKey((lambda x: (x, 1)),  # 新元素创建键值对
#                                  (lambda accSum, accVal: (accSum[0] + accVal, accSum[1] + 1)),    # 老元素累加
#                                  (lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])))    # 汇总各分区
# res = sumCount.mapValues(lambda acc: 1.0 * acc[0] / acc[1])
# res.foreach(lambda x: print(x))

# 连接操作 join
# heroRDD = sc.parallelize([('孙悟空', '大圣'), ('猪八戒', '元帅'), ('沙僧', '和尚'), ('唐僧', '三藏'), ('白骨精', '妖怪')])
# armsRDD = sc.parallelize([('孙悟空', '金箍棒'), ('猪八戒', '九齿钉耙')])
# innerRDD = heroRDD.join(armsRDD)    # inner join
# leftRDD = heroRDD.leftOuterJoin(armsRDD)    # left out join
# rightRDD = heroRDD.rightOuterJoin(armsRDD)    # right out join

# 数据排序 sortByKey
# heroRDD = sc.parallelize([('孙悟空', '大圣'), ('猪八戒', '元帅'), ('沙僧', '和尚'), ('唐僧', '三藏'), ('白骨精', '妖怪')])
# heroRDD_sorted1 = heroRDD.sortByKey(ascending=True, numPartitions=None, keyfunc=lambda x: str(x))    # 按照 key 升序排序
# heroRDD_sorted2 = heroRDD.sortByKey(ascending=True, numPartitions=None, keyfunc=lambda x: str(x)[-1])    # 按照 key 的尾字母排序

# pairRDD的行动操作 countByKey、collectAsMap、lookup
# pairRDD4: Union[RDD, Any] = sc.parallelize({(1, 2), (3, 4), (3, 6)})
# # 对每个键对应的元素分别计数
# res = pairRDD4.countByKey()
# print(res)

# print(" ")
