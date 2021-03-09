import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pre_processing import sentence_processing
from src.pre_processing import lower_first_letter
from src.tokenization import tokenization
from src.question_classifier import conf
from src.tokenization import read_stoplist
torch.manual_seed(1)
'''
input:
tokens: token list
threshold: the token whose count is below threshold will be delete.
this function is used to generate vectors of the corpus
output: the vectors of words
'''
def randomly_initialised_vectors(tokens=None,threshold=None):
    wordCountDict = dict(zip(*np.unique(tokens, return_counts=True)))
    for k in list(wordCountDict.keys()):  # 对字典a中的keys，相当于形成列表list
        if wordCountDict[k] < threshold:
            del wordCountDict[k]
    wordToIx = {}
    wordToIx['UNK'] = 0
    i = 1
    for key in wordCountDict.keys():
        wordToIx[key] = i
        i = i+1
    wordVectors = []
    dimension = int(conf.get("param","word_embedding_dim"))
    for _ in wordToIx:
        wordVectors.append(np.random.random(dimension))
    return np.array(wordVectors),wordToIx

if __name__ == '__main__':
    labels, sentences = sentence_processing(conf.get('param', 'path_train'))

    sentences = lower_first_letter(sentences)

    read_stoplist = read_stoplist()

    tokens, token_of_sentences = tokenization(sentences, read_stoplist)
    # idx
    wordVec, wordToIdx = randomly_initialised_vectors(tokens, threshold=0)
    print(wordVec)
