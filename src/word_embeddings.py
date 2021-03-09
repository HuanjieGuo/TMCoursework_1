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
    # embeds = nn.Embedding(len(wordToIx), int(conf.get("param","word_embedding_dim")))  # 2 words in vocab, 100 dimensional embeddingsr
    wordVectors = []
    dimension = int(conf.get("param","word_embedding_dim"))
    for _ in wordToIx:
        wordVectors.append(np.random.random(dimension))
    # for key in wordToIx:
    #     lookup_tensor = torch.tensor([wordToIx[key]], dtype=torch.long)
    #     embed = embeds(lookup_tensor)
    #     wordVectors.append(embed[:, :].tolist()[0])
    return np.array(wordVectors),wordToIx
