import numpy as np
from src.global_value import conf
from src.preprocessing import sentence_processing
from src.tokenization import lower_first_letter,read_stoplist,tokenization
import torch
from src.word_embeddings import get_word_embedding
from src import global_value as gv

'''
Use bag of word to sum all the vector of the tokens in one sentence and devide the count of token.

input:
tokens : a token list of a sentence
wordToIdx: a map that the key is token, and value is the idx of token in wordVec
wordVec: vectors matrix [n,vector_dimension] 

return :
vector_of_sentence: a vector that can represent the sentence.
'''
def make_bow_vector(tokens,wordToIdx,wordVec):
    vec = np.zeros(int(conf.get("param","word_embedding_dim")))
    count = 0
    for word in tokens:
        if word in wordToIdx.keys():
            vector = wordVec[wordToIdx[word]]
            vec += vector
            count+=1
    vec = vec / count
    vec = torch.from_numpy(vec)
    return vec.view(1, -1)

'''
Use a for iteration to call the 'make_bow_vector' function and append its result to a list

input:
sentences: a list of sentence
wordToIdx: a map that the key is token, and value is the idx of token in wordVec
wordVec: vectors matrix [n,vector_dimension] 

return:
sentences_vector_list: a list of sentences' vector
'''
def multi_sentences_to_vectors(sentences,wordToIdx,wordVec):
    list = []
    for tokens in sentences:
        list.append(make_bow_vector(tokens,wordToIdx,wordVec))
    return list

'''
choose what kind of vector you want to get and it will return you train, dev and test data

input:
type: randomly or pre_train
freeze: True or False

return:
train_sentence_vectors
train_labels
dev_sentence_vectors
dev_labels
test_sentence_vectors
test_labels
'''
def bag_of_word_sentences(type='randomly',freeze=True):
    if type not in ['randomly','pre_train']: return
    train_labels, train_sentences = sentence_processing(conf.get('param', 'path_train'))
    dev_labels, dev_sentences = sentence_processing(conf.get('param', 'path_dev'))
    test_labels, test_sentences = sentence_processing(conf.get('param', 'path_test'))

    train_sentences = lower_first_letter(train_sentences,conf.get('param','lowercase'))
    test_sentences = lower_first_letter(test_sentences,conf.get('param','lowercase'))
    dev_sentences = lower_first_letter(dev_sentences,conf.get('param','lowercase'))

    read_stop = read_stoplist()

    train_tokens, train_token_of_sentences = tokenization(train_sentences, read_stop)
    dev_tokens, dev_token_of_sentences = tokenization(dev_sentences, read_stop)
    test_tokens, test_token_of_sentences = tokenization(test_sentences, read_stop)
    wordVec, wordToIdx = get_word_embedding(tokens=train_tokens, type=type, freeze=freeze, path='../to_be_merged/train_1000.txt')

    train_sentence_vectors = multi_sentences_to_vectors(train_token_of_sentences,wordToIdx,wordVec)
    test_sentence_vectors = multi_sentences_to_vectors(test_token_of_sentences,wordToIdx,wordVec)
    dev_sentence_vectors = multi_sentences_to_vectors(dev_token_of_sentences,wordToIdx,wordVec)

    train_labels,dev_labels,test_labels = get_label_number_to_idx(train_labels,dev_labels,test_labels)

    return train_sentence_vectors,train_labels,dev_sentence_vectors,dev_labels,test_sentence_vectors,test_labels

'''
transform label form string to corresponding index

input:
train_labels
dev_labels
test_labels

return:
train_labels_idxs : label index
dev_labels_idxs
test_labels_idxs
'''
def get_label_number_to_idx(train_labels, dev_labels, test_labels):
    label_to_ix = {}
    for label in train_labels + test_labels+dev_labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)
    gv.label_to_ix = label_to_ix
    train_labels_idxs = []
    dev_labels_idxs = []
    test_labels_idxs = []
    for label in train_labels:
        train_labels_idxs.append(label_to_ix[label])
    for label in dev_labels:
        dev_labels_idxs.append(label_to_ix[label])
    for label in test_labels:
        test_labels_idxs.append(label_to_ix[label])
    return train_labels_idxs,dev_labels_idxs,test_labels_idxs

if __name__ == '__main__':
    bag_of_word_sentences()


