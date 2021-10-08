"""
Created on Sat Oct  2 18:24:56 2021
This program is used to calculate the recitation rate, the average edit distance and so on
 of the inference result of the data in the codegpt.

"""
import editdistance
import pymongo
import re
from fuzzywuzzy import fuzz

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["rnn-5"]
test_col = db["test"]

#cnt = 1
#limit = 1000000
flag = 0.0
flag1 = 0.0
total_db = 0.0

for item in test_col.find():
    if "best_simi_code_dis" not in item:
        test_col.delete_one(item)
        #print("delete")
        pass
    else:
        total_db += 1
        if item['best_simi_code_dis'] == 0 :
            flag += 1
            if item['ground_truth_dis'] != 0:
                flag1 += 1
            else:
                pass
        else:
            pass
        #cnt += 1
    #if cnt == limit:
        #break    

recitation = round(flag/total_db, 2)
recitation1 = round(flag1/total_db, 2)
print("flag:", flag)
print("flag1:", flag1)
print("total:", total_db)
print("recitation rate:", recitation)
print("recitation rate(remove):", recitation1)


flag2 = 0.0
gt_ave_dis = 0.0
simi_ave_dis = 0.0
for item in test_col.find():
    gt_ave_dis += item['ground_truth_dis']
    simi_ave_dis += item['best_simi_code_dis']
    if item['ground_truth_dis'] == 0:
        flag2 += 1
    else:
        pass

final_gt_ave_dis = round(gt_ave_dis/total_db, 2)
final_simi_ave_dis = round(simi_ave_dis/total_db, 2)

print("flag2:", flag2)
print("true:", round(flag2/total_db, 2))
print("final_gt_ave_dis", final_gt_ave_dis)
print("final_simi_ave_dis", final_simi_ave_dis)



def post_process(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code

EM = 0.0
edit_sim = 0.0
  
predictions_line = open("C:/Users/v-weixyan/Desktop/char-rnn/5/test_pred.txt","r",encoding="utf-8")
predictions_line_list = predictions_line.readlines()
gt_line = open("C:/Users/v-weixyan/Desktop/char-rnn/test_data/test_gt.txt","r",encoding="utf-8")
gt_line_list = gt_line.readlines()
total = len(gt_line_list)
cnt = 0
for predictions_line_list, gt_line_list in zip(predictions_line_list, gt_line_list):
    pred = post_process(predictions_line_list.strip())
    gt = post_process(gt_line_list.strip())
    edit_sim += fuzz.ratio(pred, gt)
    if pred.split() == gt.split():
        EM += 1
    #print(edit_sim)
    cnt += 1
    #print(cnt)

Final_EM = round(EM/total*100, 2)
Final_edit_sim = round(edit_sim/total, 2)
print(edit_sim)
print(EM)
print(total)
print("Final_EM", Final_EM)
print("Final_edit_sim", Final_edit_sim)