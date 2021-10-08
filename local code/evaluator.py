"""
Created on Sat Oct  2 18:24:56 2021
This program is used to calculate the edit distance and perfect match rate of the inference result of data in codegpt.

"""


from fuzzywuzzy import fuzz
import re

def post_process(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code

EM = 0.0
edit_sim = 0.0
  
predictions_line = open("C:/Users/v-weixyan/Desktop/predictions_line.txt","r",encoding="utf-8")
predictions_line_list = predictions_line.readlines()
gt_line = open("C:/Users/v-weixyan/Desktop/gt_line.txt","r",encoding="utf-8")
gt_line_list = gt_line.readlines()
total = len(gt_line_list)
cnt = 0
for predictions_line_list, gt_line_list in zip(predictions_line_list, gt_line_list):
    pred = post_process(predictions_line_list.strip())
    gt = post_process(gt_line_list.strip())
    edit_sim += fuzz.ratio(pred, gt)
    if pred.split() == gt.split():
        EM += 1
    print(edit_sim)
    cnt += 1
    print(cnt)

Final_EM = round(edit_sim/total, 2)
Final_edit_sim = round(EM/total*100, 2)
print("Final_EM", Final_EM)
print("Final_edit_sim", Final_edit_sim)