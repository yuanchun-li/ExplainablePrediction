# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to classify the types of recitation codes, which are "function call", "logical judgement", 
"mathematical operation", "function definition".

@author: v-weixyan
"""

import json
import operator
from collections import Counter
import re
import numpy as np  
import seaborn as sns  
import matplotlib.pyplot as plt 
import pandas as pd


def code_tokenize(line):
    tokens = re.findall(r"[^ \t\n\r\f\v_\W]+|[^a-zA-Z0-9_ ]", line)
    return tokens

logic_word = ["if", "elif", "else", "and", "or", "not", "in", "for",
              "break", "continue", "exit", "while", "is", "pass", "try"]
operator_word = ["+", "-", "*", "/", "%", "**", "//"]

type1 = 0
type2 = 0
type3 = 0
type4 = 0
words = []
result = []
punctuation=['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', "'", '[', ']', '@', '{', '}', '=', '+', '-',
             '&', '$', '*', '<', '>', ',', '|', '#', "%"]
with open('C:/Users/v-weixyan/Desktop/train_data_find.json', 'r', encoding="utf-8") as f:
    for jsonstr in f.readlines():
        item = json.loads(jsonstr)
        code = code_tokenize(item["code"])
        words.extend(code)
        if operator.contains(item["code"], ".") :
            type1 += 1
        if len([x for x in logic_word if x in code]) != 0:
            type2 += 1
        if len([y for y in operator_word if y in code]) != 0:
            type3 += 1
        if operator.contains(item["code"], "def") :
            type4 += 1
for i in words:
    if i not in punctuation:
        result.append(i)
result.sort()
#print(result)

result_dict = dict(Counter(result))
items = list(result_dict.items())
items.sort(key=lambda x:x[1],reverse=True)
for i in range(30):
    word,count = items[i]
    print("{0:<10}{1:>5}".format(word,count))


print("function call", type1, "\n",
      "logical judgement", type2, "\n", 
      "mathematical operation", type3, "\n",
      "function definition", type4, "\n")  

dist = [type1, type2, type3, type4]
txt = ["function call", "logical judgement", "mathematical operation", "function definition"]
a = list(result_dict.keys())
b = list(result_dict.values())
#print(a)
#print(b)
plt.bar(range(len(dist)), dist,color='b',tick_label=txt)
plt.show()

'''
import json


with open('C:/Users/v-weixyan/Desktop/train123.json', 'r', encoding="utf-8") as f:
    for jsonstr in f.readlines():
        item = json.loads(jsonstr)
        with open('C:/Users/v-weixyan/Desktop/123456.json', 'a') as json_file:
            mydict = {"id":item["id"], "code":item["recitation_code"],
                      "count":item["count"], "code_len":item["code_len"], "type":item["type"]}
            json_str = json.dumps(mydict)
            json_file.write(json_str)
            json_file.write("\n")

'''





