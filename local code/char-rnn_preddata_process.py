# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to convert the pred file format of char-rnn.

"""

import json

cnt = 1
limit = 10010
 
file = open("C:/Users/v-weixyan/Desktop/test_pred.json", 'r', encoding = 'utf-8')

for line in file.readlines():
    dic = json.loads(line)    
    code = dic["input"].split("\n")[0].strip(" ")
    id = dic["id"]
    with open("C:/Users/v-weixyan/Desktop/test_pred.txt","a") as f:
        #f.write(str(id))
        #f.write("   ")
        f.write(code)
        f.write("\n")
    if cnt == limit:
        break
    cnt += 1
