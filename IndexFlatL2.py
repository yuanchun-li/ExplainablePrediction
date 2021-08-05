import random
import torch
import numpy as np
import pymongo
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import faiss
from datasets import load_dataset

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["code_search_net"]
coordinates_col = db["coordinates"]
train_col = db["train"]
val_col = db["val"]
test_col = db["test"]

# Load stored coordinates
coordinates = coordinates_col.find_one()['coordinates']

# Load texts and stored neurons
all_texts = []
all_neurons = []
for item in train_col.find():
    all_texts.append(item['func_code_string'])
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
    input_sequences = tokenizer(text=input, add_special_tokens=True, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    input_sequences = {k: v for k,v in input_sequences.items()}
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
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

dataset = load_dataset("code_search_net", "all")
testset = dataset['test']
test_input = testset[0]['func_code_string']

sampled_neurons = sample_activations(model, tokenizer, test_input, 300)
sampled_neurons = np.array(sampled_neurons, dtype=np.float32).reshape(-1, 300)


d = 300
index = faiss.IndexFlatL2(d)
index.add(all_neurons)
print(index.ntotal)

k = 1                         
D, I = index.search(sampled_neurons, k) 
print("d",D)
print("i",I)
print(all_texts[I[0][0]])
