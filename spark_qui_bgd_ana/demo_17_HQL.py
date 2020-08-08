# -*- coding: utf-8 -*-
# @Time    : 2020/5/31 17:57
# @Author  : wangxg

from pyspark.sql import HiveContext, SparkSession

sparkSession = SparkSession.builder.master("local[*]").appName("My App").getOrCreate()
hiveCtx = HiveContext(sparkSession)
sql = "select * from dw.ods_wer_computer_use_time_scr limit 100"
rows = hiveCtx.sql(sql)
print(rows.collect)