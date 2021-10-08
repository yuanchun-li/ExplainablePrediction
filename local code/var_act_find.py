# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program uses the var_max method to select some key neurons.

"""

import json
import numpy as np
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["train_activations_01"]
train_col = db["train"]

db1 = client["var_train_activations"]
train_col_save = db1["train"]
np.set_printoptions(threshold=10000)
n_neuron = 100
'''
with open('C:/Users/v-weixyan/Desktop/py150/train_input_activations.json', 'r', encoding="utf-8") as f:
    for jsonstr in f.readlines():
        item = json.loads(jsonstr)
        tmp = []
        for i in item["activations"]:
            if i > 0:
                tmp.append(1)
            else:
                tmp.append(0)
        #print(len(tmp))
        mydict = {"rec_id":item["rec_id"], "recitation_code":item["recitation_code"], 
                  "input_id":item["input_id"], "input":item["input"], "activations": item["activations"], 
                  "act0_1":tmp
                  }
        train_col.insert_one(mydict)
'''
all_var = []
for i in range(0,9216): 
    tmp = []
    for item in train_col.find():
        tmp.append(item["act0_1"][i])
    neuron_var = np.var(np.array(tmp))
    all_var.append(neuron_var)
#print(all_var)
max_ind = np.argpartition(np.array(all_var), -n_neuron)[-n_neuron:]
top_k = np.array(all_var)[max_ind]
    #print(top_k)

for item in train_col.find():
    activations = []
    for i in max_ind:
        activations.append(item["activations"][i])
    mydict = {"rec_id":item["rec_id"], "recitation_code":item["recitation_code"], 
              "input_id":item["input_id"], "input":item["input"], "activations": activations
              }
    #print(mydict)
    train_col_save.insert_one(mydict)

