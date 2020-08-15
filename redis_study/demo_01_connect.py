# -*- coding: utf-8 -*-
# @Time    : 2020/8/12 22:57
# @Author  : wangxg

import time

import numpy as np
import redis

r = redis.Redis(host='192.168.198.135', port=6379, db=0)
keys = ['a', 'b', 'c', 'd', 'e']
values = ['马克思', '伽利略', '王阳明', '朱熹', '牛顿', '爱因斯坦']


while True:
    now = int(time.time())
    random_index_key = np.random.randint(len(keys))
    random_index_value = np.random.randint(len(values))
    random_int = np.random.randint(100000)
    r.zadd(keys[random_index_key], mapping={values[random_index_value] + '_' + str(random_int): now})
    print(keys[random_index_key], values[random_index_value] + '_' + str(random_int), now)
