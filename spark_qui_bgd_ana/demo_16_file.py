# -*- coding: utf-8 -*-
# @Time    : 2020/5/31 16:16
# @Author  : wangxg
import null as null
from pyspark import SparkContext, SparkConf
import json

conf = SparkConf().setMaster("local").setAppName("My App")
sc = SparkContext(conf=conf)
# 读取文本文件1 textFile
lines = sc.textFile("file:///usr/spark/README.md", minPartitions=3)    # 从本地读
lines1 = sc.textFile("/user/hive/warehouse/dw/ods_wer_computer_use_time_scr/ods_wer_computer_use_time_scr.txt", minPartitions=3)    # 从HDFS上读，默认

# 读取文本文件2 返回 pairRDD，(文件名,数据）
lines2 = sc.wholeTextFiles("/user/hive/warehouse/dw/ods_wer_computer_use_time_scr/ods_wer_computer_use_time_scr.txt", minPartitions=3)    # 从HDFS上读，默认
lines3 = sc.wholeTextFiles("file:///usr/spark/conf/*.template")    # 使用通配符去读取目录下某一类文件

# 保存文本文件 saveAsTextFile
# lines1.saveAsTextFile("/tmp/saveAsTextFile_test.txt")    # 保存到 HDFS 中

# 读取json文件(先当作普通文本读入，再对数据进行处理)，待改正
# jsonInput = {"items_custom_get_response":{"items":{"item":[{"num_iid":1,"product_id":0,"skus":[{"created":null,"modified":null,"outer_id":null,"price":null,"properties":null,"properties_name":"黑色","quantity":"2","sku_id":null}]}]}}}
# jsonRDD = sc.parallelize(jsonInput)
# json_data = json.load(jsonInput)
# def load(x):
#     try:
#         json.loads(x)
#     except:
#         pass
# data = lines.map(lambda x: load(x))
# print(data.collect())

# 保存json文件
# data.filter(lambda x: x["lovesOandas"]).map(lambda x: json.dumps(x)).saveAsTextFile(outputFile)

# 读取 CSV/TSV，方式同json
# 保存 CSV/TSV

# 读取hadoop格式， SequenceFile文件
# 保存hadoop格式， SequenceFile文件

# 读取对象文件，
# 保存对象文件，saveAsPickleFile
# python中，操作对象文件使用 saveAsPickleFIle() 与 pickleFile() 方法




