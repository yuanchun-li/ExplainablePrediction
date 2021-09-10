import torch
import torch.nn.functional as F
import pymongo
import re
import math
from fuzzywuzzy import fuzz
from transformers import AutoModelWithLMHead, AutoTokenizer

client = pymongo.MongoClient("mongodb://localhost:27017/")
db1 = client["re_py150_testdata"]
test_col = db1["test"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_next_token(input_ids):
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    next_token_logits = logits[0, -1, :]
    next_token_logits[2954] = -float('Inf')
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=0, top_p=0.9)
    next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
    return next_token_id

def post_process(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code


def generate(context, input_lines):
    print('++++ Generate ++++')
    #eol = context.count("<EOL>")
   # print(eol)
    #input_len = math.ceil(eol/2)
    #print(input_len)
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

total = 10000
EM = 0.0
edit_sim = 0.0
cnt = 0
for item in test_col.find({},{"_id": 0, "id":1, "code":1} ).batch_size(50):
    input = item["code"].replace("<s>","").replace("</s>","")
    predict = generate(input, 9)
    predict_code = post_process(predict[0].strip())
    ground_truth_code = post_process(predict[1].strip())
    edit_sim += fuzz.ratio(predict_code, ground_truth_code)
    cnt += 1
    if predict_code.split() == ground_truth_code.split():
        EM += 1
    print(edit_sim)
    print(cnt)

Final_EM = round(edit_sim/total, 2)
Final_edit_sim = round(EM/total*100, 2)
print(Final_EM)
print(Final_edit_sim)
