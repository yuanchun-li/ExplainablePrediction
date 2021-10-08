# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:19:30 2021
This program is used to transfer the data on the hugging face to mongodb and perform some processing on the data.

"""

import random
import torch
import numpy as np
import pymongo
from transformers import AutoTokenizer, AutoModelWithLMHead
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset('code_x_glue_cc_code_completion_token', 'python')
trainset = dataset['train']

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["gpt_py1"]
train_col = db["train"]
train_col.remove()



cnt = 1
limit = 1000000
for item in trainset:
    mydict={}
    if cnt == limit:
        break
    original_code=item['code']
    
    new_code = ""
    for i in original_code:
        if i != '<s>' and i != '<EOL>' and i != '</s>':
            new_code = new_code + i + " "
        else:
            continue
    if len(new_code)==0:
        continue
    #print(new_code)
    item["code_str"]=new_code
    mydict = item
    cnt += 1
    train_col.insert_one(mydict)