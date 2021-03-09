import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from src.pre_processing import sentence_processing
from src.pre_processing import lower_first_letter
from src.tokenization import read_stoplist
from src.tokenization import tokenization
from src.question_classifier import conf
from src.sentences_to_bow import randomly_initialised_vectors
'''
本文件中，目前对initialised_word_vector进行了测试
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

# idx
wordVec, wordToIdx = randomly_initialised_vectors(tokens, threshold=5)

data = refactor(token_of_sentences, labels)
test_data = refactor(test_token_of_sentences, test_labels)

'''
分类器
目前只有一层
输入层数量为 配置文件里word_embedding_dim
输出层数量为 label的种类数
'''
class BoWClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BoWClassifier, self).__init__()
        # self.linear = nn.Linear(vocab_size, num_labels)
        self.linear = nn.Linear(int(conf.get("param","word_embedding_dim")), num_labels)
    def forward(self, vec):
        return self.linear(vec)
        # return F.log_softmax(self.linear(vec), dim=1)


'''
输入：单个句子
1. 遍历句子每个token
2. 如果token存在高频字典，vector加上这个单词对应的随机向量
3. 如果token不存在于高频字典, vector加上 #UNK 的随机向量，wordVec[0]
4. 把累加的vector除以句子中的token个数，等到最终句子的vector
输出：单个句子的向量
'''
def make_bow_vector(sentence):
    vec = np.zeros(int(conf.get("param","word_embedding_dim")))
    for word in sentence:
        if word in wordToIdx.keys():
            vector = wordVec[wordToIdx[word]]
            vec += vector
        else : vec += wordVec[0]
    vec = vec / len(sentence)

    vec = torch.from_numpy(vec)

    return vec.view(1, -1)


def make_target(label, labels):
    return torch.LongTensor([labels[label]])


VOCAB_SIZE = len(wordToIdx)
NUM_LABELS = len(set(labels))

label_to_ix = {}
for label in labels + test_labels:
    if label not in label_to_ix:
        label_to_ix[label] = len(label_to_ix)


model = BoWClassifier(NUM_LABELS)
model.double()
loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=float(conf.get("param","lr_param")))

if __name__ == '__main__':
    for epoch in range(int(conf.get("param","epoch"))):
        for instance, label in data:
            model.zero_grad()
            bow_vec = Variable(make_bow_vector(instance))
            target = Variable(make_target(label, label_to_ix))
            output = model(bow_vec)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

        # 计算准确率
        data_size = len(test_data)
        correct_num = 0
        for instance, label in test_data:
            bow_vec = Variable(make_bow_vector(instance))
            output = model(bow_vec)
            # print(output)
            pre_max_poss, index = torch.max(output, 1)
            # print('label_index: ',label_to_ix[label])
            # print('predict_index: ',int(index))
            if label_to_ix[label] == int(index):
                correct_num += 1
        print('epoch:', epoch, ' acc: ', correct_num / data_size)