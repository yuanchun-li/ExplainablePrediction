# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 14:52:06 2021

This program is used to record the activation value of neurons.
"""

import random
import torch
import numpy as np
import pymongo
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader


# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("code_search_net", "all")
trainset = dataset['train']
testset = dataset['test']
valset = dataset['validation']

def mycollator(batch):
    return {
        'func_code_string': [x['func_code_string'] for x in batch],
    }

batch_size = 16
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, collate_fn=mycollator)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=mycollator)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=mycollator)

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["net"]
coordinates_col = db["coordinates"]
coordinates_col.remove()
train_col = db["train"]
train_col.remove()
val_col = db["val"]
val_col.remove()
test_col = db["test"]
test_col.remove()

coordinates = []
#Save the coordinates of the selected neurons
activations = []
#Save the value of the neuron
layer_shapes = []
#Save the shape of each layer
layer_index = 0
#Save the current layer number
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

def extract_shapes(self, input, output):
    global layer_shapes
    layer_shapes.append(output.shape)

def extract_neurons(self, input, output):
    global coordinates
    global activations
    global layer_index

    for index in coordinates:
        if index[0] == layer_index:
            position = index[1:]
            #print(output.shape, position)
            dim = len(output.shape)
            if dim == 2:
                activations.append(output[:,position[0]].unsqueeze(-1))
            elif dim == 3:
                activations.append(output[:,position[0],position[1]].unsqueeze(-1))
            elif dim == 4:
                activations.append(output[:,position[0],position[1],position[2]].unsqueeze(-1))
    layer_index += 1

def sample_activations(model, tokenizer, input, n_neurons):
    input_sequences = tokenizer(text=input, add_special_tokens=True, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    input_sequences = {k: v.to(device) for k,v in input_sequences.items()}
    input_sequences['output_hidden_states'] = True

    layers = get_layers(model)

    #First inference, extracts shape information of each layer
    global first_run_flag
    if first_run_flag:
        handles = []
        for l in layers:
            handle = l.register_forward_hook(extract_shapes)
            handles.append(handle)

        global layer_shapes
        layer_shapes = []
        with torch.no_grad():
            outputs = model(**input_sequences)

        # Unregister
        for handle in handles:
            handle.remove()

        global coordinates
        coordinates = []
        for i in range(n_neurons):
            cur_index = []
            cur_layer_index = random.randint(0, len(layer_shapes)-1)
            cur_index.append(cur_layer_index)

            for j in range(1, len(layer_shapes[cur_layer_index])):
                tmp = random.randint(0, layer_shapes[cur_layer_index][j]-1)

                cur_index.append(tmp)
            coordinates.append(cur_index)

        '''
        for i, s in enumerate(layer_shapes):
            print(i, s)
        print(coordinates)
        '''

        for l in layers:
            l.register_forward_hook(extract_neurons)

    # Second inference, extracts the value of n_neurons of neurons
    global activations
    activations = []
    global layer_index
    layer_index = 0
    with torch.no_grad():
        outputs = model(**input_sequences)

    return activations

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

cnt = 0
limit = 1
'''
for item in train_loader:
    if cnt == limit:
        break

    sampled_neurons = sample_activations(model, tokenizer, item['func_code_string'], 300)
    sampled_neurons = torch.cat(sampled_neurons, -1).detach().cpu().numpy()

    if first_run_flag:
        first_run_flag = False
        mydict = {'coordinates': coordinates}
        coordinates_col.insert_one(mydict)

    for i in range(sampled_neurons.shape[0]):
        mydict = {'func_code_string': item['func_code_string'][i], 'activations': sampled_neurons[i,:].tolist()}
        train_col.insert_one(mydict)

    cnt += 1

'''
'''
for item in val_loader:
    sampled_neurons = sample_activations(model, tokenizer, item['func_code_string'], 300)
    sampled_neurons = torch.cat(sampled_neurons, -1).detach().cpu().numpy()

    if first_run_flag:
        first_run_flag = False
        mydict = {'coordinates': coordinates}
        coordinates_col.insert_one(mydict)

    for i in range(sampled_neurons.shape[0]):
        mydict = {'func_code_string': item['func_code_string'][i], 'activations': sampled_neurons[i,:].tolist()}
        train_col.insert_one(mydict)
'''
for item in test_loader:
    if cnt == limit:
        break
    
    sampled_neurons = sample_activations(model, tokenizer, item['func_code_string'], 300)
    sampled_neurons = torch.cat(sampled_neurons, -1).detach().cpu().numpy()

    if first_run_flag:
        first_run_flag = False
        mydict = {'coordinates': coordinates}
        coordinates_col.insert_one(mydict)

    for i in range(sampled_neurons.shape[0]):
        mydict = {'func_code_string': item['func_code_string'][i], 'activations': sampled_neurons[i,:].tolist()}
        train_col.insert_one(mydict)
    
    cnt += 1

