import numpy as np
from src.global_value import conf
from src.preprocessing import sentence_processing
from src.tokenization import lower_first_letter,read_stoplist
import torch
from src.tokenization import tokenization
from src.word_embeddings import get_word_embedding

'''
输入：句子矩阵 n*token
1. 遍历句子每个token
2. 如果token存在高频字典，vector加上这个单词对应的随机向量
3. 如果token不存在于高频字典, vector加上 #UNK 的随机向量，wordVec[0]
4. 把累加的vector除以句子中的token个数，等到最终句子的vector
输出：多个句子的向量
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

def multi_sentences_to_vectors(sentences,wordToIdx,wordVec):
    list = []
    for tokens in sentences:
        list.append(make_bow_vector(tokens,wordToIdx,wordVec))
    return list
'''
本文件中，目前对initialised_word_vector进行了测试
'''
def refactor(sen, labels):
    data = []
    for i in range(0, len(labels)):
        data.append((sen[i], labels[i]))
    return data


def bag_of_word_sentences(type='randomly',freeze=True):
    if type not in ['randomly','pre_train']: return
    train_labels, train_sentences = sentence_processing(conf.get('param', 'path_train'))
    dev_labels, dev_sentences = sentence_processing(conf.get('param', 'path_dev'))
    test_labels, test_sentences = sentence_processing(conf.get('param', 'path_test'))

    train_sentences = lower_first_letter(train_sentences)
    test_sentences = lower_first_letter(test_sentences)
    dev_sentences = lower_first_letter(dev_sentences)

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

def get_label_number_to_idx(train_labels, dev_labels, test_labels):
    label_to_ix = {}
    for label in train_labels + test_labels+dev_labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)
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


