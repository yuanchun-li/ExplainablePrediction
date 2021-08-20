import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("code_search_net", "python")
trainset = dataset['train']
testset = dataset['test']
devset = dataset['validation']

def mycollator(batch):
    return {
        'func_code_string': [x['func_code_string'] for x in batch],
    }

batch_size = 1
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=mycollator)

def code_generate(model, tokenizer, input, len_input, len_output):
    input_sequences = tokenizer(text=input, add_special_tokens=False, padding='max_length', truncation=True, return_tensors='pt', max_length=1024)
    input_sequences = {k: v.to(device) for k,v in input_sequences.items()}
    input_sequences['output_hidden_states'] = True
    
    with torch.no_grad():
        outputs = model(**input_sequences)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    print('prediction:', prediction.cpu().item())

    
model_name = "microsoft/CodeGPT-small-py-adaptedGPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
model.eval()

cnt = 0
limit = 10
for item in test_loader:
    if cnt == limit:
        break

    code_generated = code_generate(model, tokenizer, item['func_code_string'], 100, 100)

    for i in trainset:
            recite = i["func_code_string"].find(code_generated)
            if recite != -1:
                mydict = {'code_generated': code_generated,'code_original': i["func_code_string"]}
                print(recite)
                print(mydict)
            elif recite == -1:
                pass

    cnt += 1
    print(cnt)
