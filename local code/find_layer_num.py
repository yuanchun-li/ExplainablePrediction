# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to calculate the coordinates selected by the var_max method and the hierarchical distribution of the coordinates.

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
print(max_ind)
top_k = np.array(all_var)[max_ind]
    #print(top_k)

coordinates = []
cur_index = []
all_layer = [7, 16, 25, 34, 43, 52, 61, 70, 79, 88, 97, 106]
for cur_layer_index in all_layer:
        tmp = 0
        for hid in range(768):
            cur_index = [cur_layer_index, tmp]
            cur_index.append(hid)
            coordinates.append(cur_index)
print(len(coordinates))

def list_count(num_list):
    return {x: num_list.count(x) for x in set(num_list)}

true_coordinates = []
for i in max_ind    :
    true_coordinates.append(coordinates[i])
print(true_coordinates)

layer_stat = []
for i in true_coordinates:
    layer_stat.append(i[0])
print(list_count(layer_stat))