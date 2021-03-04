import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    i = 0
    for key in wordCountDict.keys():
        wordToIx[key] = i
        i = i+1
    embeds = nn.Embedding(len(wordToIx), 5)  # 2 words in vocab, 5 dimensional embeddings
    wordVectors = []
    for key in wordToIx:
        lookup_tensor = torch.tensor([wordToIx[key]], dtype=torch.long)
        embed = embeds(lookup_tensor)
        wordVectors.append(embed[:, :].tolist()[0])
    return np.array(wordVectors)


if __name__ == '__main__':
    tokens = ['what', 'product', 'did', 'Robert', 'Conrad', 'dare', 'people', 'to', 'knock', 'off', 'his', 'shoulder', 'what', 'caused', 'Titanic', 'to', 'sink', 'what', 'diamond', 'producer', 'controls', 'about']
    wordVec = randomly_initialised_vectors(tokens,threshold=2)
    print(wordVec)