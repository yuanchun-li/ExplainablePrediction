# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program uses the top_k_top_p method to predict the next line of code,
 and calculates the edit distance between the prediction result and the ground truth, 
 but there is a problem with the prediction process and cannot be used temporarily.
"""


import torch
import torch.nn.functional as F
import Levenshtein
import pymongo
import re
from transformers import AutoModelWithLMHead, AutoTokenizer

client = pymongo.MongoClient("mongodb://localhost:27017/")
db1 = client["re_py150_testdata"]
test_col = db1["test"]

db2 = client["re_py150_row"]

db3 = client["find_result"]
find_col = db3["test"]
#find_col.remove()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Merge(dict1, dict2): 
    return(dict2.update(dict1)) 

def code_tokenize(line):
    tokens = re.findall(r"[^ \t\n\r\f\v_\W]+|[^a-zA-Z0-9_ ]", line)
    return tokens

def generate_next_token(input_ids):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    next_token_logits = logits[0, -1, :]
    next_token_logits[2954] = -float('Inf')
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=0, top_p=0.85)
    next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
    return next_token_id

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
        dis = Levenshtein.distance(predication, simi_code["code"])
        item={"simi_code":simi_code["code"],"Dis":dis, "id":simi_code["id"]}
        item_list.append(item)  
    item_list = sorted(item_list, key=lambda i: i['Dis'])
    #print(item_list)
    if len(item_list) != 0:
        min_dis = item_list[0]["Dis"]
        for each in item_list:
            value = each["Dis"]
            if value == min_dis:
                simi_dict = {"best_simi_code_id":each["id"], "best_simi_code_dis":each["Dis"], "best_simi_code":each["simi_code"]}
            else:
                break  
        return simi_dict
    else:
        new_dict = {}
        return new_dict

def generate(context, input_lines):
    print('++++ Generate ++++')
    context_first_lines = '<EOL>'.join(context.split('<EOL>')[:input_lines])
    next_line = context.split('<EOL>')[:input_lines+1][-1]
    context_ids = tokenizer.encode(context_first_lines, add_special_tokens=True)
    input_ids = context_ids
    cur_len = len(input_ids)
    init_len = len(input_ids)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    while True:
        next_token_id = generate_next_token(input_ids)
        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1)
        cur_len += 1
        word = tokenizer.convert_ids_to_tokens(next_token_id.item())
        if word == '<EOL>':
            break
        if cur_len >= 100 + init_len:
            break
    original = tokenizer.decode(input_ids.squeeze(0)[:init_len])
    result = tokenizer.decode(input_ids.squeeze(0)[init_len:])
    result = result.split('<EOL>')[0].strip()
    '''
    print('==== Input ====')
    print(original)
    print('==== Next Line ====')
    print(next_line)
    print('==== Output ====')
    print(result)
    '''
    return result, next_line



def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

model_name = "C:/Users/v-weixyan/CodeGPT-small-py-adaptedGPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)
model.eval()
model = model.to(device)


for item in test_col.find({},{"_id": 0, "id":1, "code":1} ):
    input = item["code"].replace("<s>","").replace("</s>","")
    predict = generate(input, 9)
    #print("1", predict)
    predict_code = post_process(predict[0].strip())
    #print("2", predict_code)
    predict_code = code_tokenize(predict_code)
    predict_code = " ".join(predict_code)
    #print("3", predict_code)
    predict_code = predict_code.replace("< s >", "<s>") .replace("< / s >", "</s>") 
    ground_truth_code = post_process(predict[1].strip())
    ground_truth_code = code_tokenize(ground_truth_code)
    ground_truth_code = " ".join(ground_truth_code)
    dis = Levenshtein.distance(predict_code, ground_truth_code)
    ground_truth_dict = {"id":item["id"], 
                         "predict_code":predict_code,
                         "ground_truth_code":ground_truth_code,
                         "ground_truth_dis":dis
                         }
    simi_dict = find_code(predict_code)
    Merge(ground_truth_dict, simi_dict)
    print(simi_dict)
    #find_col.insert_one(simi_dict)
