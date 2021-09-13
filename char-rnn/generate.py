#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import json
import torch
import os
import argparse

from helpers import *
from model import *


def generate(decoder, prime_str='A', predict_len=200, temperature=0.8, cuda=True):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()  # use temperature
        top_i = torch.multinomial(output_dist, 1)[0]

        # output_dist = output.data.view(-1).exp()
        # # top_i = torch.multinomial(output_dist, 1)[0]
        # top_i = output_dist.argmax()

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        #print(predicted_char)
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()
    #print(prime_str)
    new_predicted = predicted.replace(prime_str, "").split("<EOL>")[0].strip(" ")
    #print(predicted.replace(prime_str, "").split("<EOL>")[0])
    return new_predicted


# Run as standalone script
if __name__ == '__main__':
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    #argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=200)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    if args.cuda:
        decoder = torch.load(args.filename)
    else:
        decoder = torch.load(args.filename, map_location=torch.device('cpu'))
    del args.filename
    
    with open('/home/v-weixyan/Desktop/ExplainablePrediction/char-rnn/train_input.json', 'r', encoding="utf-8") as f:
        for jsonstr in f.readlines():
            item = json.loads(jsonstr)
            pred = generate(decoder, item["input"], **vars(args))
            with open("/home/v-weixyan/Desktop/ExplainablePrediction/char-rnn/train_pred.txt","a") as f:
                f.write(pred)
                f.write("\n")
