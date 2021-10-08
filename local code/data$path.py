# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to correspond the data and data path of the py150 data set, and divide the data into rows.

"""


import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["new_gptdata_py"]
train_col = db["dev"]
train_col.remove()

path = open("path.txt","r",encoding="utf-8")
path_list = path.readlines()
print(len(path_list))
code = open("code.txt","r",encoding="utf-8")
code_list = code.readlines()
print(len(code_list))
for i in range(1,len(code_list)+1):
    #mydict = {"id":str(i),"code":code_list[i-1],"path":path_list[i-1]}
    new_code = code_list[i-1].replace("<s>", "").replace("</s>", "").replace("<EOL>", "")
    mydict = {"id":str(i), 
              "code":code_list[i-1].replace("\n", ""), 
              "path":path_list[i-1].replace("\n", ""), 
              "code_str":new_code.replace("\n", "")
              }
    #print(new_code) 
    #print(mydict)
    train_col.insert_one(mydict)
''' 
code=open("code.txt","r",encoding="utf-8")
code_list=code.readlines()
for i,each in enumerate(code_list):
    s=each.split("<EOL>")
    count=1
    for block in s:
        d={"id":str(i+1)+"-"+str(count),"code":block+"<EOL>"}
        count+=1
        print(d)
'''  
'''
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
    
    #item["code_str"]=new_code
    mydict = item
    cnt += 1'''   