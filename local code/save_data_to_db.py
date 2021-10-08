# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to dump the code_search_net dataset to mongodb.

"""

import pymongo
from datasets import load_dataset

dataset = load_dataset("code_search_net", "python")
trainset = dataset['train']

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["re_code_search_net_python"]
train_col = db["train"]
train_col.remove()

cnt = 1
limit = 3000
for item in trainset:
    mydict={}
    if cnt == limit:
        break
    item["id"]=str(cnt)
    mydict=item
    cnt += 1