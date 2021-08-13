import Levenshtein
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db1 = client["new_gptdata_py_row_dis"]
train_col = db1["dev"]
train_col.remove()

code = open("code.txt", "r", encoding="utf-8")
code_list = code.readlines()

test_list = []
for i,each in enumerate(code_list[0:100]):
    s = each.split("<EOL>")
    count = 1
    for block in s:
        block = block.replace("<s>", "").replace("</s>", "").replace("\n", "")
        d = {
        "id":str(i+1)+"-"+str(count),
        "code":block
        }
        count += 1
        test_list.append(d)

test_list1 = test_list
for i in test_list:
    item_list = []
    test_list = []
    for j in test_list1:
        if i["id"] != j["id"]:
            dis = Levenshtein.distance(i["code"], j["code"])
            item = {"id":i["id"]+":"+j["id"],"Dis":dis,"code":j["code"]}
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
           "dis_matrix":test_list,
           "min_dis":min_value,
           "min_dis_index":min_value_index_list,
           "min_dis_code":min_value_code_list
        }
    train_col.insert_one(mydict)




