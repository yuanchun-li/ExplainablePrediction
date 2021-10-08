# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to convert the py150 data set from txt to json.

@author: v-weixyan
"""

import json

train_line = open("C:/Users/v-weixyan/Desktop/py150/train.txt","r",encoding="utf-8")
train_line_list = train_line.readlines()

for i,each in enumerate(train_line_list):
    with open('C:/Users/v-weixyan/Desktop/py150/train.json', 'a') as json_file:
        mydict = {"id":str(i+1), "input":each, "gt":""}
        json_str = json.dumps(mydict)
        json_file.write(json_str)
        json_file.write("\n")


test_line = open("C:/Users/v-weixyan/Desktop/py150/test.txt","r",encoding="utf-8")
test_line_list = test_line.readlines()

for i,each in enumerate(test_line_list):
    with open('C:/Users/v-weixyan/Desktop/py150/test.json', 'a') as json_file:
        mydict = {"id":str(i+1), "input":each, "gt":""}
        json_str = json.dumps(mydict)
        json_file.write(json_str)
        json_file.write("\n")


dev_line = open("C:/Users/v-weixyan/Desktop/py150/dev.txt","r",encoding="utf-8")
dev_line_list = dev_line.readlines()

for i,each in enumerate(dev_line_list):
    with open('C:/Users/v-weixyan/Desktop/py150/dev.json', 'a') as json_file:
        mydict = {"id":str(i+1), "input":each, "gt":""}
        json_str = json.dumps(mydict)
        json_file.write(json_str)
        json_file.write("\n")