# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to store the input code, pred code, and gt code of the input data into the database. 
It is also used to find the recitation code in the data input.

"""
import json
import pymongo
import re

client = pymongo.MongoClient("mongodb://localhost:27017/")
db1 = client["data_input"]
test_col = db1["test"]
'''  
with open('C:/Users/v-weixyan/Desktop/CodeGPT-Py150/py150_train/train_input.json', 'r', encoding="utf-8") as f:
    for jsonstr in f.readlines():
        item = json.loads(jsonstr)
        with open("C:/Users/v-weixyan/Desktop/CodeGPT-Py150/py150_train/input.txt","a") as file:
            file.write(item["input"])
            file.write("\n")

def code_tokenize(line):
    tokens = re.findall(r"[^ \t\n\r\f\v_\W]+|[^a-zA-Z0-9_ ]", line)
    return tokens

def post_process(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code

predictions_line = open("C:/Users/v-weixyan/Desktop/CodeGPT-Py150/py150_train/train_pred.txt","r",encoding="utf-8")
predictions_line_list = predictions_line.readlines()
input_line = open("C:/Users/v-weixyan/Desktop/CodeGPT-Py150/py150_train/input.txt","r",encoding="utf-8")
input_line_list = input_line.readlines()
gt_line = open("C:/Users/v-weixyan/Desktop/CodeGPT-Py150/py150_train/train_gt.txt","r",encoding="utf-8")
gt_line_list = gt_line.readlines()
cnt = 1
for pred, input, gt in zip(predictions_line_list, input_line_list, gt_line_list):
    pred = post_process(pred)
    pred = code_tokenize(pred)
    pred = " ".join(pred)
    mydict = {"id":cnt, "input":input.replace("\n", ""), "pred":pred.replace("\n", "").strip(" "), "gt":gt.replace("\n", "")}
    cnt += 1
    test_col.insert_one(mydict)   
'''


with open('C:/Users/v-weixyan/Desktop/CodeGPT-Py150/py150_test/test_recitation.json', 'r', encoding="utf-8") as f:
    for jsonstr in f.readlines():
        item = json.loads(jsonstr)
        if item["count"] <= 10:
            myquery = { "pred": item["recitation_code"] }
            mydoc = test_col.find(myquery, { "_id": 0, "id":1, "input": 1, "pred": 1, "gt": 1 })
            count = test_col.count_documents(myquery)
            #print(count)
            if count != 1:
                with open("C:/Users/v-weixyan/Desktop/test_data_recitiation_act.json","a") as json_file:
                    for mydict in mydoc:
                        input_dict = {"rec_id":item["id"], "recitation_code":item["recitation_code"], "input_id":mydict["id"],
                                  "input":mydict["input"]}
                        json_str = json.dumps(input_dict)
                        json_file.write(json_str)
                        json_file.write("\n")
                    
