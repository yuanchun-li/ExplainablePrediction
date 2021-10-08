# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to calculate the average cosine similarity between the recitation codes and the average similarity between the non-recitation codes.

"""

import numpy as np
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db2 = client["var_train_activations"]
train_col = db2["train"]


def cos_sim(vector_a, vector_b):
    ma = np.linalg.norm(vector_a)
    mb = np.linalg.norm(vector_b)
    sim = (np.matmul(vector_a,vector_b))/(ma*mb)
    return sim


sam_cos = []
dif_cos = []
for i in train_col.find():
    sam_recitation = []
    dif_recitation = []
    tmp = i["activations"]
    input_id = i["input_id"]
    rec_id = i["rec_id"]
    for j in train_col.find():
        if j["input_id"] != input_id:
            sim = cos_sim(tmp, j["activations"])
            if j["rec_id"] == rec_id:
                sam_recitation.append(sim)
            else:
                dif_recitation.append(sim)   
        else:
            pass
    sam_avg = np.mean(sam_recitation)
    dif_avg = np.mean(dif_recitation)
    sam_cos.append(sam_avg)
    dif_cos.append(dif_avg)


all_sam_avg = np.mean(sam_cos)
all_dif_avg = np.mean(dif_cos)
print(all_sam_avg, all_dif_avg)
