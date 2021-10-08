# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:30:43 2021
This program is used to store txt data into mongodb.

"""


import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["re_py150"]
train_col = db["train"]
train_col.remove()

cnt = 1
limit = 100000000

code=open("train.txt","r",encoding="utf-8")
code_list=code.readlines()
for item in enumerate(code_list):
    #print(item)
    #print(type(item))
    mydict = {"id":item[0], "code":item[1]}
    if cnt == limit:
        break
    cnt += 1
    train_col.insert_one(mydict)
    