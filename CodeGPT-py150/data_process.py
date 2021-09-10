import math
import json

train_line = open("C:/Users/v-weixyan/Desktop/py150/train.txt","r",encoding="utf-8")
train_line_list = train_line.readlines()

for i,each in enumerate(train_line_list):
    eol_cnt = each.count("<EOL>")
    if eol_cnt >= 5: 
        input_len = math.ceil(eol_cnt/2)
        next_line = each.split('<EOL>')[:input_len+1][-1]
        ls = next_line.split(" ")
        next_line_len = len(ls)
        if next_line_len >= 2 and next_line_len <= 50 :
            input_len = math.ceil(eol_cnt/2)
            with open('C:/Users/v-weixyan/Desktop/train_input.json', 'a') as json_file:
                context_lines = '<EOL>'.join(each.split('<EOL>')[:input_len])
                input = context_lines + str("<EOL>")
                mydict = {"id":str(i+1), "input":input, "gt":""}
                json_str = json.dumps(mydict)
                json_file.write(json_str)
                json_file.write("\n")
            with open("C:/Users/v-weixyan/Desktop/train_gt.txt","a") as f:
                f.write(next_line)
                f.write("\n")
        else:
            pass
    else:
        pass
