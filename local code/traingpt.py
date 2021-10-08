# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 16:05:02 2021
This program is used for text completion tasks, but the result is incorrect and needs to be modified.

"""


import torch
import Levenshtein
import pymongo
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.utils.data import DataLoader
import re

def code_tokenize(line):
    tokens = re.findall(r"[^ \t\n\r\f\v_\W]+|[^a-zA-Z0-9_ ]", line)
    return tokens

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["train_row"]
my_col = db["train"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("code_search_net", "python")
trainset = dataset['train']
testset = dataset['test']
devset = dataset['validation']


def mycollator(batch):
    return {
        'func_code_string': [x['func_code_string'] for x in batch],
    }

batch_size = 1
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=mycollator)

def code_generate(model, tokenizer, input, len_input):
    #print(input)
    input_sequences = tokenizer(text=input, add_special_tokens=False, padding='max_length', truncation=True, return_tensors='pt', max_length=len_input)
    input_sequences = {k: v.to(device) for k,v in input_sequences.items()}
    input_sequences['output_hidden_states'] = True
    #print(input_sequences)

    with torch.no_grad():
        outputs = model(**input_sequences)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    prediction = prediction.cpu().numpy().tolist()[0]
    #print(prediction)
    tok = DecodeIds(prediction)
    #print(tok)
    if "</s>" in tok:
        ind = tok.find("</s>")
        tok = tok[0:ind]
    print('prediction:',tok)
    return tok

def DecodeIds(idxs):
    codes = ""
    for idx in idxs:
        to_add = tokenizer.convert_ids_to_tokens(idx)
        if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
            if not codes.endswith(" "):
                codes += " " + to_add[1:]
            else:
                codes += to_add[1:]
        elif (
                idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                        tokenizer.pad_token_id] or
                tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT>")
        ):
            codes += " " + to_add + " "
        else:
            codes += to_add
    return codes.strip(" ")

def find_code(predication):
    myquery = db.train.find({"$text": { "$search": predication}}).sort([("score", {"$meta": 'textScore'})]).limit(10)   
    for simi_code in myquery:
        print("simi_code:", simi_code["code"])
        dis = Levenshtein.distance(predication, simi_code["code"])
        print("edit_dis:",dis)

        
model_name = "microsoft/CodeGPT-small-py-adaptedGPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
model.eval()

cnt = 0
limit = 3
for item in test_loader:
    if cnt == limit:
        break
    list_code = [str(i) for i in item['func_code_string']] 
    input = ' '.join(list_code) 
    
    #print(item['func_code_string'])
    #print(type(item['func_code_string']))
    input = '\n'.join(input.split('\n')[:3])
    code_generated = code_generate(model, tokenizer, input, 1024)
    found_code = find_code(code_generated)
    cnt += 1
    #print(cnt)



'''
    for i in testset:
        if code_generated in i["func_code_string"]:
            mydict = {'code_generated': code_generated,'code_original': i["func_code_string"]}
            print(mydict)
        else:
            pass
'''
   