# -*- coding: utf-8 -*-
# @Time    : 2020/7/19 17:33
# @Author  : wangxg


from multiprocessing.pool import Pool
from time import sleep


def fun(a):
    sleep(5)
    print(a)


if __name__ == '__main__':
    p = Pool(3)  # 最多执行5个进程，打印5个数
    for i in range(10):
        p.apply_async(fun, args=(i,))
    p.close()
    p.join()  # 等待所有子进程结束，再往后执行
    print("end")