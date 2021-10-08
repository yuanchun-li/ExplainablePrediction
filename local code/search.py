# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 17:14:18 2021

This program uses L2 in faiss to find the nearest neighbor vector, which is the GPU version.
"""
import random
import torch
import numpy as np
import pymongo
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import faiss


# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["code_search_net"]
coordinates_col = db["coordinates"]
train_col = db["train"]
val_col = db["val"]
test_col = db["test"]

# Load stored coordinates
coordinates = coordinates_col.find_one()['coordinates']
print(coordinates)

# Load texts and stored neurons
all_texts = []
all_neurons = []
for item in train_col.find():
    all_texts.append(item['func_code_string'])
    all_neurons.append(item['activations'])
for item in val_col.find():
    all_texts.append(item['func_code_string'])
    all_neurons.append(item['activations'])
for item in test_col.find():
    all_texts.append(item['func_code_string'])
    all_neurons.append(item['activations'])
all_neurons = np.array(all_neurons, dtype=np.float32)
print(all_neurons.shape)

# Build index
dim = 300   # 向量维度
nlist = 30 # 聚类中心的个数
quantizer = faiss.IndexFlatL2(300)
gpu_resources = faiss.StandardGpuResources()
index = faiss.GpuIndexIVFFlat(gpu_resources, dim, nlist, faiss.METRIC_L2)
# 训练
index.train(all_neurons)
index.add(all_neurons)

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
    input_sequences = tokenizer(text=input, add_special_tokens=True, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
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

model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

sampled_neurons = sample_activations(model, tokenizer, ['self.accum_param.addInPlace(self._value, term)'], 300)
sampled_neurons = np.array(sampled_neurons, dtype=np.float32).reshape(-1, 300)
print(sampled_neurons.shape)

k = 1  # 定义召回向量个数
# I表示相似index矩阵, D表示距离矩阵
D, I = index.search(sampled_neurons, k)
print(D, I)
print(all_texts[I[0][0]])




