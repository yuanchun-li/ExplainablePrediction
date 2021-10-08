# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:30:43 2021
This program is used to split the py150 data set into rows.

"""

import pymongo
import re

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["re_py150_row"]
train_col = db["train"]
train_col.remove()

def post_process(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code

def code_tokenize(line):
    tokens = re.findall(r"[^ \t\n\r\f\v_\W]+|[^a-zA-Z0-9_ ]", line)
    return tokens

code=open("train.txt","r",encoding="utf-8")
code_list=code.readlines()
for i,each in enumerate(code_list):
    each = each.replace("<s>", "").replace("</s>", "").replace("\n", "")
    s = each.split("<EOL>")
    count = 1
    for block in s:
        block = post_process(block.strip())
        block = code_tokenize(block)
        block = " ".join(block)
        mydict = {"id":str(i+1)+"-"+str(count),"code":block}
        count += 1
        #print(mydict)
        train_col.insert_one(mydict)

db.train.create_index([("code", "text")])  