# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 17:00:53 2021
This program is used to determine the type of recitation code, but because it is mutually exclusive, 
it is temporarily not used, and the method is not perfect.

"""

import json
import operator
import pymongo
import re

client = pymongo.MongoClient("mongodb://localhost:27017/")
db1 = client["re_py150_new_testdata_result"]
test_col = db1["test"]

db2 = client["re_py150_row"]
train_col = db2["train"]

def code_tokenize(line):
    tokens = re.findall(r"[^ \t\n\r\f\v_\W]+|[^a-zA-Z0-9_ ]", line)
    return tokens

logic_word = ["if", "elif", "else", "and", "or", "not", "in", "for",
              "break", "continue", "exit", "while", "is", "pass", "try"]
operator_word = ["+", "-", "*", "/", "%", "**", "//"]

cnt = 1
for item in test_col.find():
    if item["best_simi_code_dis"] == 0:
        type = 0
        myquery = { "code": item["best_simi_code"] }
        count = train_col.count_documents(myquery)
        code_len = len(item["best_simi_code"])
        code = code_tokenize(item["best_simi_code"])
        if operator.contains(item["best_simi_code"], ".") :
            type = 1
        elif len([x for x in logic_word if x in code]) != 0:
            type = 2
        elif len([y for y in operator_word if y in code]) != 0:
            type = 3
        else:
            type = 4
        with open('C:/Users/v-weixyan/Desktop/test111.json', 'a') as json_file:
            mydict = {"id":item["id"], "recitation_code":item["best_simi_code"], 
                      "gt_code":item["ground_truth_code"], "gt_dis":item["ground_truth_dis"],
                      "count":count, "code_len":code_len, "type":type}
            json_str = json.dumps(mydict)
            json_file.write(json_str)
            json_file.write("\n")
        print(cnt)
        cnt += 1
    else:
        pass
