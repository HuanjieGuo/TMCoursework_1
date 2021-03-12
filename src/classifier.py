#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 15:43:52 2021

"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from src.preprocessing import process_train_set,process_new_dataset
from src import word2vec,preprocessing
#from Bilstm_test import eval,train_another_new,acc_,train_loader,test_loader

class BiLSTMTagger(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        """
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                           bidirectional=True)

        self.fc = nn.Linear(hidden_dim * 2, 50)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        #_pad, _len = rn.pad_packed_sequence(x, batch_first=True)
        embedding = self.dropout(self.embedding(x))
        embedding = embedding.permute(1,0,2)
        output, (hidden, cell) = self.rnn(embedding)

        # [num_layers*2, b, hid_dim] => 2 of [b, hid_dim] => [b, hid_dim*2]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)

        hidden = self.dropout(hidden)
        out = self.fc(hidden)

        return out,cell
def padding_feature(int_word,seq_length):
   
    '''
    :param sentence2idx:
    :return: feature of sentences, if the length of sentences is less than seq_length,padding with 0
    else truncated to the seq_length
    '''
    feature  = np.zeros((len(int_word),seq_length),dtype = int)
    for i,row in enumerate(int_word):
        feature[i,-len(row):] = np.array(row)[:seq_length]

    assert len(feature) == len(int_word), "Your features should have as many rows as reviews."
    assert len(feature[0]) == seq_length, "Each feature row should contain seq_length values."
    return feature

def to_dataloader(feature,labels):
    labels = np.array(labels)
    #print(labels)
    train_data = TensorDataset(torch.from_numpy(feature), torch.from_numpy(labels))
    # dataloaders
    batch_size = 1

    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    return train_loader


def vector_file():

    train_int_word,train_int_label,word2idx,lable2idx = process_train_set('../data/train.txt')
    dev_int_word,dev_int_label = process_new_dataset(word2idx,lable2idx,'../data/dev.txt')
    test_int_word, test_int_label = process_new_dataset(word2idx, lable2idx, '../data/test.txt')


    device = 'cpu'
    model = torch.load('../data/word2vec_Bilstm_3.pkl')
    model.to(device)

    seq_length = 10
    feature_metrix = padding_feature(train_int_word,seq_length)
    feature_metrix_dev = padding_feature(dev_int_word,seq_length)
    feature_metrix_test = padding_feature(test_int_word, seq_length)

    train_loader = to_dataloader(feature_metrix,train_int_label)
    dev_loader = to_dataloader(feature_metrix_dev,dev_int_label)
    test_loader = to_dataloader(feature_metrix_test, test_int_label)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = optim.SGD(rnn_.parameters(), lr=1e-3)
    criteon = nn.CrossEntropyLoss().to(device)

    with open('../data/train_.txt','w',encoding='utf-8') as f:
        for inputs, labels in train_loader:
            pred,cell = model(inputs.long())
            rep = cell.view(200,1).squeeze(1)
            sen_rep =rep.data.numpy().tolist()
            f.write(str(sen_rep))
            f.write(str(int(labels)))
            f.write('\n')

    with open('../data/dev_.txt','w',encoding='utf-8') as f:
        for inputs, labels in dev_loader:
            pred,cell = model(inputs.long())
            rep = cell.view(200,1).squeeze(1)
            sen_rep =rep.data.numpy().tolist()
            f.write(str(sen_rep))
            f.write(str(int(labels)))
            f.write('\n')

    with open('../data/test_.txt','w',encoding='utf-8') as f:
        for inputs, labels in test_loader:
            pred,cell = model(inputs.long())
            rep = cell.view(200,1).squeeze(1)
            sen_rep =rep.data.numpy().tolist()
            f.write(str(sen_rep))
            f.write(str(int(labels)))
            f.write('\n')
