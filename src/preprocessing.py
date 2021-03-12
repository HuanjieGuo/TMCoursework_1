# By Maxine

from collections import Counter
from src import word2vec

# 1 remove all the '?' of each sentence
# 2 split every line into 2 lists : label and sentences
# 3 lowercase the first letter of every sentence
# 4 将sentences进行tokenization并去除没有用的符号与词（初步停用词）

'''
整个预处理文件的使用：
// conf.get('param', 'path_train')表示传入的txt文件
1. labels, sentences = sentenceProcessing(conf.get('param', 'path_train'))
2. sentences = lower_first_letter(sentences,conf.get('param','lowercase'))
'''

from src.global_value import conf

'''
sentenceProcessing方法
输入：txt文件
输出：两个list，labels, sentences
作用：将文件的每一行划分成独立的标签和句子
使用格式：labels, sentences = sentenceProcessing(conf.get('param', 'path_train'))
'''


def sentence_processing(file):
    labels = []
    sentences = []
    with open(file, 'r') as f:
        # result = f.read()
        # result = re.sub('[?]','',result)
        for line in f.readlines():
            line = line.strip('\n')
            # print(line)
            # print(line.split(' ',1))
            labels.append(line.split(' ', 1)[0])
            sentences.append(line.split(' ', 1)[1])
    return labels, sentences


'''
lower_first_letter方法：
输入：list，包含了多个句子
输出：小写化句子首字母的list
作用：小写化句子首字母
'''


def lower_first_letter(sentences, isLower=True):
    if isLower:
        for i in range(len(sentences)):
            tmp = list(sentences[i])
            tmp[0] = tmp[0].lower()
            sentences[i] = ''.join(tmp)
    return sentences


'''
预处理：
'''

if __name__ == '__main__':
    labels, sentences = sentence_processing(conf.get('param', 'path_train'))
    sentences = lower_first_letter(sentences, conf.get('param','lowercase'))

    print(len(labels))
    print(len(sentences))


def process_train_set(path, lowercase=True):
    with open(path, 'r') as f:
        data = f.readlines()
        labels, sentences = labels_extraction(data)
        sentences = remove_punctuations(sentences)
        if lowercase:
            sentences = to_lower_case(remove_stop_words(sentences))
        sorted_words = make_vocabulary(sentences)
        word_idx, _ = word2vec.create_dict(sorted_words)
        sentences_in_idx = word2vec.replace_words_with_idx(sentences, word_idx)
        label_idx = get_label_idx(labels)
        labels_in_idx = replace_labels_with_idx(labels, label_idx)
        return sentences_in_idx, labels_in_idx, word_idx, label_idx


def process_new_dataset(word_idx, label_idx, path, lowercase=True):
    with open(path, 'r') as f:
        data = f.readlines()
        labels, sentences = labels_extraction(data)
        sentences = remove_punctuations(sentences)
        if lowercase:
            sentences = to_lower_case(remove_stop_words(sentences))
        labels_in_idx = []
        for label in labels:
            if label in label_idx:
                labels_in_idx.append(label_idx[label])
            else:
                label_idx[label] = len(label_idx)
                labels_in_idx.append(label_idx[label])
        sentences_in_idx = []
        for sentence in sentences:
            sentence_idx = []
            for word in sentence:
                if word in word_idx:
                    sentence_idx.append(word_idx[word])
            sentences_in_idx.append(sentence_idx)
        return sentences_in_idx, labels_in_idx


def labels_extraction(data):
    labels = []
    sentences = []
    for line in data:
        s = line.split()
        labels.append(s[0])
        sentences.append(s[1:])
    return labels, sentences


def get_label_idx(labels):
    label_list = list(set(labels))
    label_list.append('unknown_')
    return {label_list[i]: i for i in range(len(label_list))}


def replace_labels_with_idx(labels, label_idx):
    return [label_idx[labels[i]] for i in range(len(labels))]


def remove_punctuations(sentences):
    for sentence in sentences:
        i = 0
        while i < len(sentence):
            if sentence[i] in "?''``,.&...":
                sentence.remove(sentence[i])
            else:
                i += 1
    return sentences


def to_lower_case(sentences):
    for sentence in sentences:
        sentence[0] = sentence[0].lower()
    return sentences


def remove_stop_words(sentences):
    stop_words = []
    with open('../data/stop_words.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            stop_words.append(line.strip('\n'))
    for sentence in sentences:
        i = 0
        while i < len(sentence):
            if sentence[i] in stop_words:
                sentence.remove(sentence[i])
            else:
                i += 1
    return sentences


def make_vocabulary(sentences):
    all_words = []
    for sentence in sentences:
        all_words.extend(sentence)
    word_frequency = Counter(all_words)
    sorted_words = sorted(word_frequency, key=word_frequency.get, reverse=True)
    vocabulary = sorted_words
    return vocabulary


def get_preprocessed_sentences(lowercase=True):
    with open('../data/train.txt', 'r') as f:
        data = f.readlines()
        _, sentences = labels_extraction(data)
        sentences = remove_punctuations(sentences)
        sentences = remove_stop_words(sentences)
        if lowercase:
            return to_lower_case(sentences)
        else:
            return sentences


# print(get_preprocessed_sentences())


# if __name__ == '__main__':
    # with open('./train.txt', 'r') as f:
    #     data = f.readlines()
    #     labels, sentences = labels_extraction(data)
    #     sentences = remove_punctuations(sentences)
    #     sentences = to_lower_case(sentences)
    #     # sentences = remove_stop_words(sentences)
    #     vocabulary = make_vocabulary(sentences)
    #     with open('vocabulary.txt', 'w') as fv:
    #         for word in vocabulary[:-1]:
    #             fv.write(word + '\n')
    #         fv.write(vocabulary[-1])
    # sentences_in_idx, labels, word_idx = process_train_set('../data/train.txt')
    # train_int_word, train_int_label, word2idx, lable2idx = process_train_set('../data/train.txt')

    '''
    print(len(sentences_in_idx))
    print(len(labels))
    print(len(word_idx))
    
    s = 'How many points make up a perfect fivepin bowling score ?'
    sentences_in_idx = process_single_sentence(s,word2idx)
    print(list(lable2idx.keys())[list(lable2idx.values()).index(2)])
    #print(sentences_in_idx)
    print(type(lable2idx))
    '''
