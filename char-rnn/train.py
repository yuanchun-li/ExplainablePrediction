#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import editdistance
import re

from tqdm import tqdm

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=5000)
argparser.add_argument('--print_every', type=int, default=200)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--save_path', type=str, default="/home/v-weixyan/Desktop/ExplainablePrediction/char-rnn/py150.pt")
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

file, file_len = read_file(args.filename)
train_len = int(file_len * 0.8)


def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, train_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def code_tokenize(line):
    tokens = re.findall(r"[^ \t\n\r\f\v_\W]+|[^a-zA-Z0-9_ ]", line)
    return tokens

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data.item() / args.chunk_len


def save():
    if args.save_path is None:
        save_path = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    else:
        save_path = args.save_path
    dir_path = os.path.dirname(save_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    torch.save(decoder, save_path)
    print('Saved as %s' % save_path)


# Initialize models and start training
decoder = CharRNN(
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0
train_loss = 0.0
train_acc = 0.0
x = []
y = []
train_loss_list = []
train_acc_list = []

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(args.chunk_len, args.batch_size))
        loss_avg += loss
        x.append(epoch)
        
        train_loss_list.append(loss)
        
        if epoch % args.print_every == 0:
            plt.figure(figsize=(20, 12), dpi=100)
            plt.subplot(2, 1, 1)
            try:
                train_loss_lines.remove(train_loss_lines[0]) 
            except Exception:
                pass

            train_loss_lines = plt.plot(x, train_loss_list, 'b', lw=1) 
            plt.title("loss")
            plt.xlabel("epoch")
            plt.ylabel("")
            plt.legend("train_loss")

            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            input = '<s> def comparer ( ) : <EOL> a = <NUM_LIT:3> <EOL> b = <NUM_LIT:4> <EOL>'
            pred = generate(decoder, input, 200, cuda=args.cuda)
            new_pred = pred.replace(input, "").split("<EOL>")[0].strip(" ")
            gt ='for i in range ( <NUM_LIT> ) :'
            gt = gt.strip(" ")
            pred_arr = code_tokenize(new_pred)
            gt_arr = code_tokenize(gt)
            dis = editdistance.eval(pred_arr, gt_arr)
            print(new_pred, '\n', gt, dis)
            
            y.append(epoch)
            train_acc_list.append(dis)
            plt.subplot(2, 1, 2)
            try:
                train_acc_lines.remove(train_acc_lines[0]) 
            except Exception:
                pass

            train_acc_lines = plt.plot(y, train_acc_list, 'b', lw=1) 
            plt.title("dis")
            plt.xlabel("epoch")
            plt.ylabel("")
            plt.legend("train_dis")
            
            if epoch % 500 == 0:
                plt.savefig('/home/v-weixyan/Desktop/ExplainablePrediction/char-rnn/save'+str(epoch)+'.png')
            
            plt.ion()
            plt.pause(3)
            plt.close()
    with open("/home/v-weixyan/Desktop/ExplainablePrediction/char-rnn/loss.txt","a") as f:
        x_str = ', '.join([str(i) for i in x])
        train_loss_list_str = ', '.join([str(i) for i in train_loss_list])
        f.write(x_str)
        f.write("\n")
        f.write(train_loss_list_str)
    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()
