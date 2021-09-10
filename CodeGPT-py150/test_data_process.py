import math
import json
import random
import linecache
random.seed(42)
list_info = []
while True:
    info = random.randint(1,46451)
    if info not in list_info:
        list_info.append(info)
    if len(list_info) == 15000:
        break

def getline(the_file_path, line_number):
  if line_number < 1:
    return ''
  for cur_line_number, line in enumerate(open(the_file_path, 'rU')):
    if cur_line_number == line_number-1:
      return line
  return ''

cnt = 0
limit = 10000
for i in enumerate(list_info):
    the_line = linecache.getline('C:/Users/v-weixyan/Desktop/py150/test.txt', i[1])
    eol_cnt = the_line.count("<EOL>")
    if eol_cnt >= 5: 
        input_len = math.ceil(eol_cnt/2)
        next_line = the_line.split('<EOL>')[:input_len+1][-1]
        ls = next_line.split(" ")
        next_line_len = len(ls)
        if next_line_len >= 2 and next_line_len <= 50 :
            input_len = math.ceil(eol_cnt/2)
            with open('C:/Users/v-weixyan/Desktop/py150_test/test_input.json', 'a') as json_file:
                context_lines = '<EOL>'.join(the_line.split('<EOL>')[:input_len])
                input = context_lines + str("<EOL>")
                mydict = {"id":str(i[1]), "input":input, "gt":""}
                json_str = json.dumps(mydict)
                json_file.write(json_str)
                json_file.write("\n")
            with open("C:/Users/v-weixyan/Desktop/py150_test/test_gt.txt","a") as f:
                f.write(next_line)
                f.write("\n")
            cnt += 1
        else:
            pass
    else:
        pass
    if cnt == limit:
        break
