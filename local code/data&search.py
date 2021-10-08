"""
Created on Sat Oct  2 18:24:56 2021

This program is used to find the most similar line of code in mongodb.

"""


import Levenshtein
import pymongo
import re

def code_tokenize(line):
    tokens = re.findall(r"[^ \t\n\r\f\v_\W]+|[^a-zA-Z0-9_ ]", line)
    return tokens

client = pymongo.MongoClient("mongodb://localhost:27017/")
db1 = client["re_code_x_glue_cc_code_completion_line_python"]
test_col = db1["train"]

db2 = client["code_x_glue_cc_code_completion_line_python_new"]
my_col = db2["train"]
my_col.remove()


code_list = []
for item in test_col.find({},{"_id": 0, "id":1, "code":1} ):
    code_list.append(item)
#print(code_list)   

for i,each in enumerate(code_list):
    s = each["code"].replace("<s>","").replace("</s>","").split("<EOL>")
    count = 1
    for block in s:
        if block != "" and block != "    ''''''" and block != '    """' and block != '        """' and  block != '    """"""' and block != "'''":
            block = block.strip(" ")
            block = code_tokenize(block)
            block = " ".join(block)
            d = {
                "id":str(i+1)+"-"+str(count),
                "code":block.strip(" ")
                }
            my_col.insert_one(d)
            count += 1
               
db2.test.create_index([("code", "text")])               

def Merge(dict1, dict2): 
    return(dict2.update(dict1)) 

new_code = "code_list.append(item)"
print("input code:",new_code)

def find_code(predication):
    predication = code_tokenize(predication)
    predication = " ".join(predication)
    myquery = db2.test.find({"$text": { "$search": predication}}).sort([("score", {"$meta": 'textScore'})]).limit(100)   
    item_list=[]
    for simi_code in myquery:
        #print("simi_code:", simi_code["code"])
        dis = Levenshtein.distance(predication, simi_code["code"])
        item={"simi_code":simi_code["code"],"Dis":dis, "id":simi_code["id"]}
        item_list.append(item)
       # print("edit_dis:",dis)   
    item_list = sorted(item_list, key=lambda i: i['Dis'])
    min_dis = item_list[0]["Dis"]
    for each in item_list:
        value = each["Dis"]
        if value == min_dis:
            simi_dict = {"best_simi_code_id":each["id"], "best_simi_code_dis":each["Dis"], "best_simi_code":each["simi_code"]}
            # print(simi_dict)
        else:
            break  
    return simi_dict
print(find_code(new_code))


#found_code = find_code(new_code)
