# -*- coding: utf-8 -*-
# @Time    : 2020/8/13 22:57
# @Author  : wangxg


import redis

r = redis.StrictRedis(host='192.168.198.135', port=6379)
print(r.dbsize())
print(r.keys())
print(r.set("小明", 11))
print(r.keys())
print(r.get("小明"))
