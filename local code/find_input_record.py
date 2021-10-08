# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:46:56 2021
This program used to compute the new input and use faiss to compute the nearest neighbour.
@author: v-weixyan
"""

import torch
import numpy as np
import pymongo
from transformers import AutoTokenizer, AutoModelWithLMHead
import faiss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["gpt_row_record"]
coordinates_col = db["coordinates"]
train_col = db["train"]


# Load stored coordinates
coordinates = coordinates_col.find_one()['coordinates']

# Load texts and stored neurons
all_texts = []
all_neurons = []
for item in train_col.find():
    all_texts.append(item['code'])
    all_neurons.append(item['activations'])
all_neurons = np.array(all_neurons, dtype=np.float32)
print(all_neurons.shape)


activations = []
#save the value of the neuron
layer_shapes = []
#save the shape of each layer
layer_index = 0
#save the current layer number
first_run_flag = True

def get_layers(model):
    layers = []

    def unfold_layer(model):
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)

            if sublayer_num == 0:
                layers.append(module)
            elif isinstance(module, torch.nn.Module):
                unfold_layer(module)

    unfold_layer(model)

    return layers

def extract_neurons(self, input, output):
    global coordinates
    global activations
    global layer_index

    for index in coordinates:
        if index[0] == layer_index:
            position = index[1:]
            dim = len(output.shape)
            if dim == 2:
                activations.append(output[:,position[0]].unsqueeze(-1))
            elif dim == 3:
                activations.append(output[:,position[0],position[1]].unsqueeze(-1))
            elif dim == 4:
                activations.append(output[:,position[0],position[1],position[2]].unsqueeze(-1))
    layer_index += 1

def sample_activations(model, tokenizer, input, n_neurons):
    input_sequences = tokenizer(text=input, add_special_tokens=True, padding='max_length', truncation=True, return_tensors='pt', max_length=514)
    input_sequences = {k: v.to(device) for k,v in input_sequences.items()}
    input_sequences['output_hidden_states'] = True

    layers = get_layers(model)

    # only register hook at the first time
    global first_run_flag
    if first_run_flag:
        for l in layers:
            l.register_forward_hook(extract_neurons)
        first_run_flag = False

    # second inference, extracts the value of n_neurons of neurons
    global activations
    activations = []
    global layer_index
    layer_index = 0
    with torch.no_grad():
        outputs = model(**input_sequences)

    return activations

model_name = "microsoft/CodeGPT-small-py-adaptedGPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
model.eval()



sampled_neurons = sample_activations(model, tokenizer, 'import functools', 10000)
sampled_neurons = np.array(sampled_neurons, dtype=np.float32).reshape(-1, 10000)



d = 10000
index = faiss.IndexFlatL2(d)
index.add(all_neurons)
print(index.ntotal)

k = 3                        
D, I = index.search(sampled_neurons, k) 
print("d",D)
print("i",I)
print(type(I))
for i in range(0,k):
    print(all_texts[I[0][i]])



 



