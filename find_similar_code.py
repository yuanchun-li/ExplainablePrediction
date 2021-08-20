import Levenshtein
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")
db1 = client["re_code_search_net_python"]
test_col = db1["test"]

db2 = client["code_search_net_row"]
my_col = db2["test"]


code_list = []
for item in test_col.find({},{"_id": 0, "id":1, "func_code_string":1} ):
    code_list.append(item)
    

for i,each in enumerate(code_list):
    s = each["func_code_string"].split("\n")
    count = 1
    for block in s:
        if block != "" and block != "    ''''''" and block != '    """' and block != '        """' and  block != '    """"""':
            d = {
                "id":str(i+1)+"-"+str(count),
                "code":block
                }
            my_col.insert_one(d)
            count += 1
               
db2.test.create_index([("code", "text")])               

new_code = "coordinates = coordinates_col.find_one()['coordinates']"
print("input code:",new_code)
def find_code(predication):
    myquery = db2.test.find({"$text": { "$search": predication}}).sort([("score", {"$meta": 'textScore'})]).limit(10)   
    for simi_code in myquery:
        print("simi_code:", simi_code["code"])
        dis = Levenshtein.distance(predication, simi_code["code"])
        print("edit_dis:",dis)
        
found_code = find_code(new_code)
