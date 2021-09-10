#!/usr/bin/env python

import os
import sys
import numpy
import string
import unidecode
import argparse
import torch
import Levenshtein
import torch.nn as nn
from torch.autograd import Variable
from search import SearchEngine
from model import *


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


def predict(model, prime_str='A', predict_len=100, cuda=False):
    hidden = model.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[:,p], hidden)

    inp = prime_input[:, -1]

    predicted = ''
    for p in range(predict_len):
        output, hidden = model(inp, hidden)

        output_dist = output.data.view(-1).exp()
        # top_i = torch.multinomial(output_dist, 1)[0]
        top_i = output_dist.argmax()

        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted


def load_data(data_path):
    content = unidecode.unidecode(open(data_path, errors='ignore').read())
    total_len = len(content)
    train_len = int(total_len * 0.8)
    train_content = content[:train_len]
    test_content = content[train_len:]
    return train_content, test_content


class RecitationEvaluator(object):
    def __init__(self, model_path, data_path, input_len=1, output_len=1, use_cuda=False):
        self.model_path = model_path
        self.data_path = data_path
        self.input_len = input_len
        self.output_len = output_len
        self.use_cuda = use_cuda
        self.train_content, self.test_content = load_data(data_path)
        if use_cuda:
            model = torch.load(model_path).cuda()
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model = model
        db_name = os.path.basename(data_path)
        db_name = db_name[:db_name.rfind('.')]
        self.search_engine = SearchEngine(db_name, self.train_content)

    def evaluate(self, text_content):
        print(f'evaluating...')
        lines = text_content.splitlines()
        n_lines = len(lines)
        results = []
        for i, line in enumerate(lines):
            if i < self.input_len or i >= n_lines - self.output_len:
                continue
            if len(lines[i].strip()) <= 1:
                continue
            if len(lines[i-1].strip()) <= 1:
                continue
            input_text = '\n'.join(lines[i-self.input_len:i]) + '\n'
            output_text = '\n'.join(lines[i:i+self.output_len])
            predict_text = predict(model=self.model, prime_str=input_text,
                                   predict_len=len(output_text) * 2, cuda=self.use_cuda)
            print(f'----------------------\ninput: {lines[i-1].strip()}')
            print(f'  predicted: {predict_text.strip().splitlines()[0]}')
            comparison = self._compare(predict_text.strip(), output_text.strip(), self.output_len)
            results.append(comparison)
        return results

    def _compare(self, predict_text, output_text, output_len):
        predict_lines = predict_text.splitlines() + [''] * output_len
        predict_lines = predict_lines[:output_len]
        output_lines = output_text.splitlines()
        test_dist = Levenshtein.distance(output_lines[0], predict_lines[0]) / len(predict_lines[0])
        train_line, train_dist = self._find_similar_lines(predict_lines[0])
        print(f'  test line ({test_dist:.2f}): {output_lines[0]}')
        print(f'  train line ({train_dist:.2f}): {train_line}')
        return test_dist, train_dist

    def _find_similar_lines(self, query_line):
        r = self.search_engine.search(query_line)
        r = [(code, dist/len(code)) for code, dist in r]
        if not r:
            return 'N/A', -1
        r = sorted(r, key=lambda x:x[1])
        return r[0]


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-mp', '--model_path', type=str)
    argparser.add_argument('-dp', '--data_path', type=str)
    argparser.add_argument('-il', '--input_len', type=int, default=1)
    argparser.add_argument('-ol', '--output_len', type=int, default=1)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    evaluator = RecitationEvaluator(args.model_path, args.data_path,
                                    args.input_len, args.output_len, args.cuda)
    evaluator.evaluate(evaluator.test_content)


if __name__ == '__main__':
    main()

