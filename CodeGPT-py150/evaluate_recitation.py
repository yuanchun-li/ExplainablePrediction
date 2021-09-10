import editdistance
import pymongo
import re

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["re_py150_testdata_rusult"]
test_col = db["test"]

#cnt = 1
#limit = 1000000
flag = 0
total = 0

for item in test_col.find():
    if "best_simi_code_dis" not in item:
        test_col.delete_one(item)
        print("delete")
        pass
    else:
        total += 1
        if item['best_simi_code_dis'] ==0 :
            flag += 1
        else:
            pass
        #cnt += 1
    #if cnt == limit:
        #break    

recitation = round(flag/total, 2)
print("flag:", flag)
print("total:", total)
print("recitation rate:", recitation)

