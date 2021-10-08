# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program uses the method with the lowest activation frequency of neurons to select some key neurons.

"""

import json
import numpy as np
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db2 = client["min_train_activations"]
train_col = db2["train"]


np.set_printoptions(threshold=10000)
n_neuron = 100
with open('C:/Users/v-weixyan/Desktop/py150/train_input_activations.json', 'r', encoding="utf-8") as f:
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
    c_min = np.min(base)
    if c_min != 0:
        min_ind = np.argpartition(base,n_neuron)[:n_neuron]
    else:
        c_min_index = np.where(base == c_min)
        new_base = np.delete(base,c_min_index)
        c_min = np.min(new_base)
        coordinates = []
        for index, value in enumerate(list(base)):
            if value == c_min:
                coordinates.append(index)
            else:
                pass
        while True: 
            if len(coordinates) >= n_neuron:         
                break
            c_min_index = np.where(new_base == c_min)
            new_base = np.delete(new_base,c_min_index)
            c_min = np.min(new_base)
            for index, value in enumerate(base):
                if value == c_min:
                    coordinates.append(index)
                else:
                    pass

coordinates = coordinates[:n_neuron]
with open('C:/Users/v-weixyan/Desktop/py150/train_input_activations.json', 'r', encoding="utf-8") as f:
    for jsonstr in f.readlines():
        item = json.loads(jsonstr)
        activations = []
        for i in coordinates:
            activations.append(item["activations"][i])
        mydict = {"rec_id":item["rec_id"], "recitation_code":item["recitation_code"], 
                  "input_id":item["input_id"], "input":item["input"], "activations": activations
                  }
        train_col.insert_one(mydict)
