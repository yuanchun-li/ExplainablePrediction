import random
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


def sample_activations(model_name, input, n_neurons):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    input_sequences = tokenizer(text=input, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
    input_sequences = {k: v for k,v in input_sequences.items()}
    input_sequences['output_hidden_states'] = True
 
    layers = get_layers(model)
    for l in layers:
        l.register_forward_hook(extract)

    global activations
    activations = []

    model.eval()
    with torch.no_grad():
        outputs = model(**input_sequences)

    all_neurons = np.concatenate(activations)
    size = len(all_neurons)
    cnt = 0
    sampled_neurons = []
    sampled_assist = []
    while cnt < n_neurons:
        index = random.randint(0, size - 1)
        if index not in sampled_assist and all_neurons[index] != 0:
            sampled_assist.append(index)
            sampled_neurons.append(all_neurons[index])
            cnt += 1

    return outputs, sampled_neurons


activations = []
def extract(self, input, output):
    global activations
    if len(output.shape) == 4:
        activations.append(output[0,0,:,0].detach().cpu().numpy().flatten())
    elif len(output.shape) == 3:
        activations.append(output[0,:,0].detach().cpu().numpy().flatten())
    else:
        activations.append(output[0,0].detach().cpu().numpy().flatten())


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

outputs, sampled_neurons = sample_activations("bert-base-uncased", "Today is a nice day!", 300)

print('sampled_neurons:', sampled_neurons)

