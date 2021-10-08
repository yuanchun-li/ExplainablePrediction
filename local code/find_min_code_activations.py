# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:24:56 2021
This program is used to calculate the activation value of a specific neuron of the code with the smallest edit distance.

"""

import random
import torch
import pymongo
from transformers import AutoTokenizer, AutoModelWithLMHead
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

client = pymongo.MongoClient("mongodb://localhost:27017/")
db1 = client["gpt_row_record1"]
coordinates_col = db1["coordinates"]
coordinates_col.remove()
train_col = db1["train"]
train_col.remove()

db2 = client["new_gptdata_py_row_dis"]
my_col = db2["dev"]


def mycollator(batch):
    return {
        'id': [x['id'] for x in batch],
        'code': [x['code'] for x in batch],
        'min_dis': [x['min_dis'] for x in batch],
        'min_dis_code': [x['min_dis_code'] for x in batch],
        
    }

trainset = []
for x in my_col.find():
    trainset.append(x)
    
  
batch_size = 1    
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, collate_fn=mycollator)


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

model_name = "microsoft/CodeGPT-small-py-adaptedGPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
model.eval()


cnt = 0
limit = 20000

for item in train_loader:
    if cnt == limit:
        break

    sampled_neurons = sample_activations(model, tokenizer, item['code'], 10000)
    sampled_neurons = torch.cat(sampled_neurons, -1).detach().cpu().numpy()
    
   
    if first_run_flag:
        first_run_flag = False
        mydict = {'coordinates': coordinates}
        coordinates_col.insert_one(mydict)

    for i in range(sampled_neurons.shape[0]):
        mydict = {'id': item['id'][i],
                  'code': item['code'][i],
                  'min_dis': item['min_dis'][i],
                  'min_dis_code': item['min_dis_code'][i],
                  'activations': sampled_neurons[i,:].tolist()}
        train_col.insert_one(mydict)

    cnt += 1
    print(cnt)
    
