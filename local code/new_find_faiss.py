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
db = client["var"]
train_col = db["train"]

all_rec_id = []
all_input_id = []
all_id = []
all_texts = []
all_neurons = []
all_recitation_code = []
for item in train_col.find():
    all_id.append(item['id'])
    all_rec_id.append(item['rec_id'])
    all_input_id.append(item['input_id'])
    all_texts.append(item['input'])
    all_recitation_code.append(item["recitation_code"])
    all_neurons.append(item['activations'])
all_neurons = np.array(all_neurons, dtype=np.float32)
print(all_neurons.shape)
all_neurons = np.array(all_neurons) 

d = 100
index = faiss.IndexFlatL2(d)
index.add(all_neurons)
print(index.ntotal)

k = 6
with open("/home/v-weixyan/Desktop/var_faiss.json","a") as json_file:
    for i in range(0,len(all_id)):
        act = np.array(all_neurons[i]).reshape(-1, 100)
        D, I = index.search(act, k)
        code_input = []
        for j in range(0,k):
            code_input.append(all_texts[I[0][j]])
            #print(all_texts[I[0][j]])
        I = I + 1
        mydict = {"id":all_id[i], "rec_id":all_rec_id[i], "input_id":all_input_id[i], "code":all_texts[i], 
                  "dis":D.tolist(), "index":I.tolist(), "recitation_code":all_recitation_code[i], "input":code_input}
        json_str = json.dumps(mydict)
        json_file.write(json_str)
        json_file.write("\n")
    