#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:09:46 2021

"""
import numpy as np
from collections import Counter
import torch


# from src.preprocessing1 import process_train_set,process_new_dataset


# print(get_preprocessed_sentences())
def remove_single_punctuations(sentence):
    i = 0
    while i < len(sentence):
        if sentence[i] in "?''``,.&...":
            sentence.remove(sentence[i])
        else:
            i += 1
    return sentence


def process_single_sentence(s, word2idx, lowercase=True):
    s = s.split()
    sentences = remove_single_punctuations(s)
    if lowercase:
        sentences[0] = sentences[0].lower()
    sentences_in_idx = []
    sentences_in_idx.append([word2idx[word] for word in sentences])
    return sentences_in_idx


def get_class(sentences_in_idx, lable2idx, rnn):
    sen = torch.Tensor(sentences_in_idx).long()
    # pred = rnn(sen.unsqueeze(-1))

    pred = rnn(sen)
    preds = pred.max(dim=1, keepdim=True)[1].squeeze(1)
    sen_class = list(lable2idx.keys())[list(lable2idx.values()).index(preds)]
    return sen_class
