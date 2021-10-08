# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to select neurons in the activation function layer, but it is no longer used.


"""
import torch

from datasets import load_dataset
raw_datasets = load_dataset("imdb")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

#print("模型结构：",model)

from transformers import TrainingArguments
training_args = TrainingArguments("test_trainer")

import numpy as np
from transformers import Trainer
from datasets import load_metric

batch = small_eval_dataset[9]

batch = {k: v for k, v in batch.items()}
batch['output_hidden_states'] = True

del batch['text']
del batch['label']
batch['input_ids'] = torch.tensor(batch['input_ids'], dtype=torch.long).unsqueeze(dim=0)
batch['token_type_ids'] = torch.tensor(batch['token_type_ids'], dtype=torch.long).unsqueeze(dim=0)
batch['attention_mask'] = torch.tensor(batch['attention_mask'], dtype=torch.long).unsqueeze(dim=0)

saved_data = [] 

def printnorm(self, input, output):
    layer_data = []
    layer_data.append('层：'+self.__class__.__name__)
    layer_data.append(input[0])
    layer_data.append(output)
    saved_data.append(layer_data)

def getLayers(model):
    
    layers = []

    def unfoldLayer(model):
        
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)

            if sublayer_num == 0:
                layers.append(module)
            elif isinstance(module, torch.nn.Module):
                unfoldLayer(module)

    unfoldLayer(model)

    return layers

layers = getLayers(model)

for l in layers:
    # print("第"+str(l)+"层")
    l.register_forward_hook(printnorm)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
model = model.to(device)
model.eval()

with torch.no_grad():
     outputs = model(**batch)

import random

ActFun = []
random.seed(10)
for i in range(0,12):
    Act_Softmax=8+i*11
    b = saved_data[Act_Softmax][1].cpu().numpy()
    res = random.sample(range(0, 512), 100)
    b = b[:,:,res,:]
    ActFun.append(b)
    Act_Gelu = 13+i*11
    b = saved_data[Act_Gelu][2].cpu().numpy()
    res = random.sample(range(0, 512), 100)
    b = b[:,res,:]
    ActFun.append(b)

b = saved_data[138][2].cpu().numpy()
res = random.sample(range(0, 768), 100)
b = b[:,res]
ActFun.append(b)
print(type(ActFun))
ActFun = np.array(ActFun)
print(ActFun.shape)
