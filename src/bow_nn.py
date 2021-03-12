import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from src import sentence_vector
from src.global_value import conf
from src import global_value as gv
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
        self.f1 = nn.Linear(int(conf.get("param","word_embedding_dim")), num_labels)
        # self.f2 = nn.Linear(n_hidden, num_labels)

        self.double()
        # loss
        self.loss_function = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=float(conf.get("param", "lr_param")))

        self.test_vecs = []
        self.test_label = []
        self.label_to_ix = {}

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
        error_list = []
        for i in range(len(test_labels)):
            bow_vec = Variable(test_sentence_vectors[i])
            label = test_labels[i]
            output = self(bow_vec)

            pre_max_poss, index = torch.max(output, 1)
            if label == int(index):
                correct_num += 1
            else:
                for key in self.label_to_ix:
                    if int(index)==self.label_to_ix[key]:
                        error_list.append((i,key))
                        break
        outputErrorSentenceToFile(error_list)
        return round(correct_num / data_size,4)
def outputErrorSentenceToFile(error_list):
    f = open(gv.conf.get("param","path_test"))
    sentences = f.read().split('\n')
    f.close()
    error_sens = []
    for idx,label in error_list:
        error_sens.append((sentences[idx],label))
    error_sens = np.array(error_sens)
    fo = open("../data/error_sentences.txt", "w")
    fo.write(str(error_sens))
    fo.close()


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
            sentence_vectors.append(vec)
            labels.append(int(label))
        except:
            break
        line = f.readline()
    f.close()
    return sentence_vectors,labels

def train():
    train_sentence_vectors,train_labels,dev_sentence_vectors,dev_labels,test_sentence_vectors,test_labels = sentence_vector.bag_of_word_sentences(type=conf.get("param", "word_embedding_type"), freeze=True)

    output_size = len(set(train_labels))
    model = QuestionClassifier(output_size)
    # save test data
    model.test_vecs = test_sentence_vectors
    model.test_label = test_labels
    model.label_to_ix = gv.label_to_ix
    for epoch in range(int(conf.get("param","epoch"))):
        model.train_model(train_sentence_vectors,train_labels)
        # # 计算验证集准确率
        acc = model.test_model(dev_sentence_vectors,dev_labels)
        print('epoch:', epoch, ' dev_acc: ', acc)
    torch.save(model, conf.get("param", "path_model"))

def test():
    model = torch.load(conf.get('param','path_model'))
    model.to('cpu')

    # 计算测试集
    acc = model.test_model(model.test_vecs, model.test_label)
    print('test_acc: ', acc)


if __name__ == '__main__':
    # 修改randomly 或者pre_train 选择不同word embeding方法
    train_sentence_vectors,train_labels,dev_sentence_vectors,dev_labels,test_sentence_vectors,test_labels = sentence_vector.bag_of_word_sentences(type='pre_train', freeze=True)

    output_size = len(set(train_labels))
    model = QuestionClassifier(output_size)

    for epoch in range(int(conf.get("param","epoch"))):
        model.train_model(train_sentence_vectors,train_labels)
        # # 计算验证集准确率
        acc = model.test_model(dev_sentence_vectors,dev_labels)
        print('epoch:', epoch, ' dev_acc: ', acc)
    # 计算测试集
    acc = model.test_model(test_sentence_vectors, test_labels)
    print('test_acc: ', acc)



