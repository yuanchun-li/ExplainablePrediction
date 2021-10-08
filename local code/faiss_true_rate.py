# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program used to compute the correct rate of var_max method.

@author: v-weixyan
"""

import json
import numpy as np
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["var_data_faiss"]
train_col = db["train"]

with open('C:/Users/v-weixyan/Desktop/py150/456.json', 'r', encoding="utf-8") as f:
    right = 0
    cnt = 0
    limit = 10000
    
    num = 0
    for jsonstr in f.readlines():
        cnt += 1
        if cnt == limit:
            break
        item = json.loads(jsonstr)
        l = item["input_new_id"].split(',')
        m = [ int(x) for x in l ]
        print(m)
        if len(m) == 1:
            num += 1
            true_index = []
            for i in train_col.find():
                if i["rec_id"] == item["rec_id"]:
                    a = i["input_new_id"].split(',')
                    true_index.extend(a)
            numbers = [ int(x) for x in true_index ]
            numbers = np.unique(numbers).tolist()
            d = ["abc" for c in numbers if c in item["index"][0]]
            if len(d) >= 2:
                right += 1            
            else:
                pass
        else:
            pass
    print(right)
    print(num)
        