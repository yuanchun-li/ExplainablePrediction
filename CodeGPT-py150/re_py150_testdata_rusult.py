import editdistance
import pymongo
import re

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["re_py150_testdata_rusult"]
test_col = db["test"]
#test_col.remove()

db2 = client["re_py150_row"]  

def Merge(dict1, dict2): 
    return(dict2.update(dict1)) 

def code_tokenize(line):
    tokens = re.findall(r"[^ \t\n\r\f\v_\W]+|[^a-zA-Z0-9_ ]", line)
    return tokens

def post_process(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code

def find_code(predication):
    myquery = db2.train.find({"$text": { "$search": predication}}).sort([("score", {"$meta": 'textScore'})]).limit(100)   
    item_list=[]
    for simi_code in myquery:
        simi_code_arr = code_tokenize(simi_code["code"])
        predication_arr = code_tokenize(predication)
        dis = editdistance.eval(predication_arr, simi_code_arr)
        item={"simi_code":simi_code["code"],"Dis":dis, "id":simi_code["id"]}
        item_list.append(item)  
    item_list = sorted(item_list, key=lambda i: i['Dis'])
    #print(item_list)
    if len(item_list) != 0:
        min_dis = item_list[0]["Dis"]
        for each in item_list:
            value = each["Dis"]
            if value == min_dis:
                simi_dict = {"best_simi_code_id":each["id"], 
                             "best_simi_code_dis":each["Dis"], 
                             "best_simi_code":each["simi_code"]
                             }
            else:
                break  
        return simi_dict
    else:
        new_dict = {}
        return new_dict    

cnt = 1
limit = 10
   
predictions_line = open("C:/Users/v-weixyan/Desktop/predictions_line.txt","r",encoding="utf-8")
predictions_line_list = predictions_line.readlines()
gt_line = open("C:/Users/v-weixyan/Desktop/gt_line.txt","r",encoding="utf-8")
gt_line_list = gt_line.readlines()
for predictions_line_list, gt_line_list in zip(predictions_line_list, gt_line_list):
    if cnt == limit:
        break
    predict_code = post_process(predictions_line_list.strip())
    predict_code_arr = code_tokenize(predict_code)
    predict_code = " ".join(predict_code_arr)
    simi_dict = find_code(predict_code)
    ground_truth_code = post_process(gt_line_list.strip())
    ground_truth_code_arr = code_tokenize(ground_truth_code)
    ground_truth_code = " ".join(ground_truth_code_arr)
    dis = editdistance.eval(predict_code_arr, ground_truth_code_arr)
    mydict = {"id":str(cnt), 
              "predict_code":predict_code,
              "ground_truth_code":ground_truth_code,
              "ground_truth_dis":dis
              }
    Merge(simi_dict, mydict)
    print(mydict)
    cnt += 1
    #test_col.insert_one(mydict)
