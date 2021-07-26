import random
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

coordinates = []
#Save the coordinates of the selected neurons
activations = []
#Save the value of the neuron
layer_shapes = []
#Save the shape of each layer
layer_index = 0
#Save the current layer number

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
            position = ','.join(str(i) for i in index[1:])
            activations.append(output[eval(position)].item())
    layer_index += 1

def sample_activations(model_name, input, n_neurons):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    input_sequences = tokenizer(text=input, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
    input_sequences = {k: v for k,v in input_sequences.items()}
    input_sequences['output_hidden_states'] = True

    layers = get_layers(model)

#First inference, extracts shape information of each layer 
    for l in layers:
        l.register_forward_hook(extract_shapes)

    global layer_shapes
    layer_shapes = []
    model.eval()
    with torch.no_grad():
        outputs = model(**input_sequences)

    global coordinates
    coordinates = []
    for i in range(n_neurons):
        cur_index = []
        cur_layer_index = random.randint(0, len(layer_shapes)-1)
        cur_index.append(cur_layer_index)
        cur_index.append(0)
        
        for j in range(1, len(layer_shapes[cur_layer_index])):
            tmp = random.randint(0, layer_shapes[cur_layer_index][j]-1)

            cur_index.append(tmp)
        coordinates.append(cur_index)
        
# Second inference, extracts the value of n_neurons of neurons    
    for l in layers:
        l.register_forward_hook(extract_neurons)

    global activations
    activations = []
    global layer_index
    layer_index = 0
    model.eval()
    with torch.no_grad():
        outputs = model(**input_sequences)

    return activations

sampled_neurons = sample_activations("bert-base-uncased", "Today is a nice day!", 300)

print('sampled_neurons:', sampled_neurons)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    