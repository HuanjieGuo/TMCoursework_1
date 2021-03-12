import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from src.global_value import conf
from src import Bilstm_test1
torch.manual_seed(1)


'''
分类器
目前网络有一层隐藏层
输入层数量为 配置文件里word_embedding_dim
输出层数量为 label的种类数
'''
class QuestionClassifier(nn.Module):
    def __init__(self, num_labels):
        super(QuestionClassifier, self).__init__()
        # n_hidden = 256
        self.f1 = nn.Linear(int(conf.get("param","word_embedding_dim")), num_labels)
        # self.f2 = nn.Linear(n_hidden, num_labels)

        self.double()
        # loss
        self.loss_function = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=float(conf.get("param", "lr_param")))

    def forward(self, input):
        out = self.f1(input)
        # out = F.sigmoid(out)
        # out = self.f2(out)
        return out

    def train_model(self,sentence_vectors,labels):
        for i in range(0, len(sentence_vectors)):
            vector = sentence_vectors[i]
            label = labels[i]
            self.zero_grad()
            bow_vec = Variable(vector)
            target = Variable(torch.LongTensor([label]))

            output = self(bow_vec)

            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()

    def test_model(self,test_sentence_vectors,test_labels):
        # calculate correct rate
        data_size = len(test_sentence_vectors)
        correct_num = 0

        for i in range(len(test_labels)):
            bow_vec = Variable(test_sentence_vectors[i])
            label = test_labels[i]
            output = self(bow_vec)

            pre_max_poss, index = torch.max(output, 1)

            if label == int(index):
                correct_num += 1

        return round(correct_num / data_size,4)


def readFile(file):
    sentence_vectors = []
    labels = []

    f = open(file)
    line = f.readline()
    while line:
        try:
            list,label = line.split(']')
            list = list.lstrip('[')
            list = np.fromstring(list,dtype=float,sep=", ")
            vec = torch.from_numpy(list)
            vec = vec.view(1, -1)
            # print(vec)
            sentence_vectors.append(vec)
            labels.append(int(label))
        except:
            break
        line = f.readline()
    f.close()
    return sentence_vectors,labels


def train():
    Bilstm_test1.train_Bilstm()
    train_sentence_vectors, train_labels = readFile("./train_.txt")
    dev_sentence_vectors, dev_labels = readFile("./dev_.txt")
    test_sentence_vectors, test_labels = readFile("./test_.txt")

    output_size = len(set(train_labels))
    model = QuestionClassifier(output_size)

    model.test_vecs = test_sentence_vectors
    model.test_label = test_labels

    for epoch in range(int(conf.get("param","epoch"))):
        model.train_model(train_sentence_vectors,train_labels)
        acc = model.test_model(dev_sentence_vectors, dev_labels)
        print('epoch:', epoch, 'test_acc: ', acc)
    torch.save(model, conf.get("param", "path_model"))


def test():
    model = torch.load(conf.get('param', 'path_model'))
    model.to('cpu')

    acc = model.test_model(model.test_vecs, model.test_label)
    print('test_acc: ', acc)

    # #
    # train_sentence_vectors,train_labels,dev_sentence_vectors,dev_labels,test_sentence_vectors,test_labels = sentence_vector.bag_of_word_sentences(type='pre_train')
    # train_sentence_vectors, train_labels = readFile("./train_.txt")
    # test_sentence_vectors,test_labels = readFile("./text_.txt")

    # #
    # output_size = len(set(train_labels))
    # model = QuestionClassifier(output_size)

    # for epoch in range(int(conf.get("param","epoch"))):
    #     model.train_model(train_sentence_vectors,train_labels)
    #     acc = model.test_model(test_sentence_vectors, test_labels)
    #     print('epoch:', epoch, 'test_acc: ', acc)

    # # 计算测试集
    # acc = model.test_model(test_sentence_vectors, test_labels)
    # print('test_acc: ', acc)


if __name__ == '__main__':
    train()
    test()