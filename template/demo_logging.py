# -*- coding: utf-8 -*-
# @Time    : 2020/7/6 22:00
# @Author  : wangxg

import logging
handler_1 = logging.StreamHandler()  # 输出到控制台
handler_2 = logging.FileHandler("G:\@文件管理\【03】代码仓库\【99】python日志\logging_test1.log")  # 输出到文件
# handler_3 = logging.handlers.RotatingFileHandler("G:\@文件管理\【03】代码仓库\【99】python日志\logging_test2.log")  # 按照大小自动分割日志文件，一旦达到指定的大小重新生成文件
# handler_4 = logging.handlers.TimedRotatingFileHandler("G:\@文件管理\【03】代码仓库\【99】python日志\logging_test2.log")  # 按照时间自动分割日志文件
fmt = "%(asctime)s %(levelname)s %(filename)s %(funcName)s [line:%(lineno)d] %(message)s"
logging.basicConfig(level=logging.INFO, format=fmt, handlers=[handler_1, handler_2])

for i in range(100):
    logging.info("第%s行日志" % i)