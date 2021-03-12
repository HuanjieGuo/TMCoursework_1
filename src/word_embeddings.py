import numpy as np
import torch.nn as nn
import torch
from src import word2vec
from src.preprocessing import sentence_processing,lower_first_letter
from src.tokenization import tokenization,read_stoplist
from src.global_value import conf
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
    word_vectors = []
    for _ in wordToIx:
        word_vectors.append(np.random.random(int(conf.get("param","word_embedding_dim"))))
    word_vectors = np.array(word_vectors)
    return word_vectors,wordToIx


def get_word_embedding(tokens, type='randomly', freeze=True,path=None):
    if type == 'randomly':
        wordVec,wordToIdx =  randomly_initialised_vectors(tokens, threshold=5)
    if type == 'pre_train':
        wordVec,wordToIdx = get_pre_train_vector()

    embeds = nn.Embedding.from_pretrained(torch.from_numpy(wordVec),freeze=freeze)
    wordvec = embeds.weight.data.numpy()
    return wordvec,wordToIdx


'''
输入
拿到预训练的单词向量
'''
def get_pre_train_vector():
    print('Please wait, pre-train...')
    sentences = word2vec.preprocessing.get_preprocessed_sentences()
    sorted_words = word2vec.preprocessing.make_vocabulary(sentences)
    word_idx, idx_word = word2vec.create_dict(sorted_words)

    sentences_in_idx = word2vec.replace_words_with_idx(sentences, word_idx)
    # this
    word_to_vec = word2vec.train(len(sorted_words), int(conf.get('param', 'word_embedding_dim')), sentences_in_idx,
                                 idx_word)

    return word_to_vec, word_idx

if __name__ == '__main__':
    labels, sentences = sentence_processing(conf.get('param', 'path_train'))

    sentences = lower_first_letter(sentences,conf.get('param','lowercase'))

    read_stoplist = read_stoplist()

    tokens, token_of_sentences = tokenization(sentences, read_stoplist)
    # idx
    wordVec, wordToIdx = randomly_initialised_vectors(tokens, threshold=0)
    # print(wordVec)
