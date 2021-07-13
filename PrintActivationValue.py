import torch

from datasets import load_dataset
raw_datasets = load_dataset("imdb")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

print("模型结构：",model)

from transformers import TrainingArguments
training_args = TrainingArguments("test_trainer")

import numpy as np
from transformers import Trainer
from datasets import load_metric

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
batch = small_eval_dataset[9]

batch = {k: v for k, v in batch.items()}
batch['output_hidden_states'] = True

del batch['text']
del batch['label']
batch['input_ids'] = torch.tensor(batch['input_ids'], dtype=torch.long).unsqueeze(dim=0)
batch['token_type_ids'] = torch.tensor(batch['token_type_ids'], dtype=torch.long).unsqueeze(dim=0)
batch['attention_mask'] = torch.tensor(batch['attention_mask'], dtype=torch.long).unsqueeze(dim=0)

def printnorm(self, input, output):
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('input shape: ', input[0].shape)
    print('input value: ', input[0])
    print('output shape: ', output.shape)
    print('output value: ', output)

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
    print("====================================")
    print("第"+str(l)+"层")
    l.register_forward_hook(printnorm)

model.eval()
with torch.no_grad():
    outputs = model(**batch)

logits = outputs.logits
prediction = torch.argmax(logits, dim=-1)
print('prediction:', prediction.cpu().item())

print(len(outputs.hidden_states))
for i, activations in enumerate(outputs.hidden_states):
    print("输出的层数以及输出的形状：",i, activations.shape)



