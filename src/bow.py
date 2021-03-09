import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from src.pre_processing import sentence_processing
from src.pre_processing import lower_first_letter
from src.tokenization import read_stoplist
from src.tokenization import tokenization
from src.question_classifier import conf
from src.word_embeddings import randomly_initialised_vectors

'''
refactor(sen, labels):
作用：将数据合并为【（[]）,label】的形式

'''


def refactor(sen, labels):
    data = []
    for i in range(0, len(labels)):
        data.append((sen[i], labels[i]))
    return data


labels, sentences = sentence_processing(conf.get('param', 'path_train'))
test_labels, test_sentences = sentence_processing(conf.get('param', 'path_test'))

sentences = lower_first_letter(sentences)
test_sentences = lower_first_letter(test_sentences)

read_stoplist = read_stoplist()

tokens, token_of_sentences = tokenization(sentences, read_stoplist)
test_tokens, test_token_of_sentences = tokenization(test_sentences, read_stoplist)

data = refactor(token_of_sentences, labels)
test_data = refactor(test_token_of_sentences, test_labels)


class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        # self.linear = nn.Linear(vocab_size, num_labels)
        self.linear = nn.Linear(vocab_size, num_labels)
    def forward(self, vec):
        return self.linear(vec)
        # return F.log_softmax(self.linear(bow_vec), dim=1)



def make_bow_vector(sentence, vocab):
    vec = torch.zeros(len(vocab))
    for word in sentence:
        vec[vocab[word]] += 1
    return vec.view(1, -1)


def make_target(label, labels):
    return torch.LongTensor([labels[label]])


word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = len(set(labels))

label_to_ix = {}
for label in labels + test_labels:
    if label not in label_to_ix:
        label_to_ix[label] = len(label_to_ix)
# print(label_to_ix)

model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

if __name__ == '__main__':
    for epoch in range(10):
        for instance, label in data:
            model.zero_grad()
            bow_vec = Variable(make_bow_vector(instance, word_to_ix))
            target = Variable(make_target(label, label_to_ix))
            output = model(bow_vec)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

        # 计算准确率
        data_size = len(test_data)
        correct_num = 0
        for instance, label in test_data:
            bow_vec = Variable(make_bow_vector(instance, word_to_ix))
            output = model(bow_vec)
            # print(output)
            pre_max_poss, index = torch.max(output, 1)
            # print('label_index: ',label_to_ix[label])
            # print('predict_index: ',int(index))
            if label_to_ix[label] == int(index):
                correct_num += 1
        print('epoch:', epoch, ' acc: ', correct_num / data_size)


