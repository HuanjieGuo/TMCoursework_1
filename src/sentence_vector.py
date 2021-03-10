import numpy as np
from src.question_classifier import conf
from src.pre_processing import sentence_processing
from src.tokenization import lower_first_letter,read_stoplist
import torch
from src.tokenization import tokenization
from src.word_embeddings import word_to_vector

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
    for word in tokens:
        if word in wordToIdx.keys():
            vector = wordVec[wordToIdx[word]]
            vec += vector
        else : vec += wordVec[0]
    vec = vec / len(tokens)
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


def bag_of_word_sentences(type='randomly'):
    if type not in ['randomly','pre_train']: return
    labels, sentences = sentence_processing(conf.get('param', 'path_train'))
    test_labels, test_sentences = sentence_processing(conf.get('param', 'path_test'))

    sentences = lower_first_letter(sentences)
    test_sentences = lower_first_letter(test_sentences)

    read_stop = read_stoplist()

    tokens, token_of_sentences = tokenization(sentences, read_stop)
    test_tokens, test_token_of_sentences = tokenization(test_sentences, read_stop)
    wordVec, wordToIdx = word_to_vector(tokens=tokens,type=type,path='../to_be_merged/train_1000.txt')

    sentence_vectors = multi_sentences_to_vectors(token_of_sentences,wordToIdx,wordVec)
    test_sentence_vectors = multi_sentences_to_vectors(test_token_of_sentences,wordToIdx,wordVec)

    labels,test_labels = get_label_number_to_idx(labels,test_labels)
    return sentence_vectors,labels,test_sentence_vectors,test_labels

def get_label_number_to_idx(labels,test_labels):
    label_to_ix = {}
    for label in labels + test_labels:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)
    labels_idxs = []
    test_labels_idxs = []
    for label in labels:
        labels_idxs.append(label_to_ix[label])
    for label in test_labels:
        test_labels_idxs.append(label_to_ix[label])
    return labels_idxs,test_labels_idxs

if __name__ == '__main__':
    bag_of_word_sentences()


