#!/usr/bin/env python

import os
import sys
import numpy
import string
import unidecode
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable


all_characters = string.printable
n_characters = len(all_characters)


# Turning a string into a tensor
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-mp', '--model_path', type=str)
    argparser.add_argument('-dp', '--data_path', type=str)
    argparser.add_argument('-il', '--input_len', type=int, default=100)
    argparser.add_argument('-ol', '--output_len', type=int, default=100)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()
    return args


def load_data(data_path):
    content = unidecode.unidecode(open(data_path, errors='ignore').read())
    total_len = len(content)
    train_len = int(total_len * 0.8)
    train_content = content[:train_len]
    test_content = content[train_len:]
    return train_content, test_content


def generate(decoder, prime_str='A', predict_len=100, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)

    inp = prime_input[:, -1]

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        output_dist = output.data.view(-1).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted


def main():
    args = parse_args()
    train_content, test_content = load_data(args.data_path)
    engine = SearchEngine()

