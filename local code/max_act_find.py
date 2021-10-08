# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program uses the method with the highest activation frequency of neurons to select some key neurons.

"""

import json
import numpy as np
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db2 = client["123456789"]
train_col = db2["train"]

np.set_printoptions(threshold=10000)
n_neuron = 100
with open('C:/Users/v-weixyan/Desktop/train_input_activations.json', 'r', encoding="utf-8") as f:
    base = [0 for x in range(0,9216)]
    for jsonstr in f.readlines():
        item = json.loads(jsonstr)
        tmp = []
        for i in item["activations"]:
            if i > 0:
                tmp.append(1)
            else:
                tmp.append(0)
        base = np.array(base) + np.array(tmp)
    max_ind = np.argpartition(base, -n_neuron)[-n_neuron:]
    print(max_ind)
    top_k = base[max_ind]
    #print(top_k)

with open('C:/Users/v-weixyan/Desktop/train_input_activations.json', 'r', encoding="utf-8") as f:
    for jsonstr in f.readlines():
        item = json.loads(jsonstr)
        activations = []
        for i in max_ind:
            activations.append(item["activations"][i])
        mydict = {"rec_id":item["rec_id"], "recitation_code":item["recitation_code"], 
                  "input_id":item["input_id"], "input":item["input"], "activations": activations
                  }
        #print(mydict)
        train_col.insert_one(mydict)

