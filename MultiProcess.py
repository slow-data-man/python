# -*- coding: utf-8 -*-
# @Time    : 2020/7/19 17:32
# @Author  : wangxg


from multiprocessing.pool import Pool
from time import sleep


def fun(a):
    sleep(5)
    print(a)


if __name__ == '__main__':
    p = Pool()  # 这里不加参数，但是进程池的默认大小，等于电脑CPU的核数
    # 也是创建子进程的个数，也是每次打印的数字的个数
    for i in range(10):
        p.apply_async(fun, args=(i,))
    p.close()
    p.join()  # 等待所有子进程结束，再往后执行
    print("end")