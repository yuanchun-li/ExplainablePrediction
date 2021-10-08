# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to record the code with the smallest edit distance between codes.

"""


import Levenshtein
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db1 = client["new_gptdata_py"]
train_col = db1["dev"]

db2 = client["gpt_dis"]
my_col = db2["dev"]


test_list = []
for x in train_col.find():
    test_list.append(x)
    

#cnt = 1
test_list1 = test_list
for i in test_list:
    item_list = []
    test_list = []
    for j in test_list1:
        if i["id"] != j["id"]:
            dis = Levenshtein.distance(i["code_str"], j["code_str"])
            item = {"id":i["id"]+":"+j["id"],"Dis":dis,"code":j["code_str"]}
            item_list.append(item)
            test_list.append(dis)
    item_list = sorted(item_list, key = lambda k: k['Dis'])
    min_value = item_list[0]["Dis"]
    min_value_index_list = []
    min_value_code_list = []
    for each in item_list:
        value = each["Dis"]
        if value==min_value:
            min_value_index = each["id"]
            min_value_code = each["code"]
            min_value_index_list.append(min_value_index)
            min_value_code_list.append(min_value_code)
        else:
            break
    mydict = {"id":i["id"],
              "code":i["code"],
              "path":i["path"],
           "code_str":i["code_str"],
           "dis_matrix":test_list,
           "min_dis":min_value,
           "min_dis_index":min_value_index_list,
           "min_dis_code":min_value_code_list
        }
    #cnt += 1
    #print(mydict)
    my_col.insert_one(mydict)
    #if cnt == 10:
        #break




