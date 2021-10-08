# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program selects some key neurons by randomly selecting neurons.

"""


import json
import random
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db1 = client["random_train_activations"]
train_col = db1["train"]



first_run_flag = True
random.seed(42)
with open('C:/Users/v-weixyan/Desktop/py150/train_input_activations.json', 'r', encoding="utf-8") as f:
    random_coordinates = []
    for jsonstr in f.readlines():
        item = json.loads(jsonstr)
        #print(len(item["activations"]))
        if first_run_flag:
            first_run_flag = False
            for i in range(100):
                tmp = random.randint(0, len(item["activations"])-1)
                random_coordinates.append(tmp)
        activations = []
        for i in random_coordinates:
            activations.append(item["activations"][i])
        mydict = {"rec_id":item["rec_id"], "recitation_code":item["recitation_code"], 
                  "input_id":item["input_id"], "input":item["input"], "activations": activations
                  }
        train_col.insert_one(mydict)
    