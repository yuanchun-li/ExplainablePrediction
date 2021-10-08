# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to do the nearest neighbor search with faiss for each data.

@author: v-weixyan
"""

import torch
import numpy as np
import pymongo
import faiss
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

client = pymongo.MongoClient("mongodb://localhost:27017/")
db1 = client["data_input_var_activations"]
train_col = db1["activations"]

db2 = client["var"]
var_col = db2["train"]
 

all_input_id = []
all_texts = []
all_neurons = []

for item in train_col.find():
    all_input_id.append(item['input_id'])
    all_texts.append(item['input'])
    all_neurons.append(item['activations'])
all_neurons = np.array(all_neurons, dtype=np.float32)
print(all_neurons.shape)

d = 100
index = faiss.IndexFlatL2(d)
index.add(all_neurons)
print(index.ntotal)

k = 10
with open("/home/v-weixyan/Desktop/123.json","a") as json_file:
    for item in var_col.find():
        var_new_input_id = []
        var_neurons = []
        myquery = { "input":item['input'] }
        mydoc = train_col.find(myquery)
        for x in mydoc:
            var_new_input_id.append(x["input_id"])
        var_neurons.append(item['activations'])
        act = np.array(var_neurons, dtype=np.float32).reshape(-1, 100)
        D, I = index.search(act, k)
        code_input = []
        for j in range(0,k):
            code_input.append(all_texts[I[0][j]])
            #print(all_texts[I[0][j]])
        I = I + 1
        mydict = {"rec_id":item['rec_id'], "input_id":item['input_id'], "input_new_id":str(var_new_input_id), "index":I.tolist(), "dis":D.tolist()}
        json_str = json.dumps(mydict)
        json_file.write(json_str)
        json_file.write("\n")
        
'''         
        var_rec_id.append(item['rec_id'])
        var_recitation_code.append(item["recitation_code"])
        var_input_id.append(item['input_id'])
        var_texts.append(item['input'])
        var_neurons.append(item['activations'])

    
    
    
    
   
    for i in range(0,len(var_rec_id)):
        act = np.array(var_neurons[i]).reshape(-1, 100)
        D, I = index.search(act, k)
        code_input = []
        for j in range(0,k):
            code_input.append(all_texts[I[0][j]])
            #print(all_texts[I[0][j]])
        I = I + 1
        #mydict = {"rec_id":var_rec_id[i], "recitation_code":var_recitation_code[i], "input_id":var_input_id[i], 
        #          "input_new_id":var_new_input_id[i], "input":var_texts[i], "dis":D.tolist(), "index":I.tolist(), "input_simi":code_input}
        mydict = {"rec_id":var_rec_id[i], "input_id":var_input_id[i], "input_new_id":var_new_input_id[i], "dis":D.tolist(), "index":I.tolist()}
        json_str = json.dumps(mydict)
        json_file.write(json_str)
        json_file.write("\n")
'''