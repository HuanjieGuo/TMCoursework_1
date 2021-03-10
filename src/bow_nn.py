import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from src.question_classifier import conf
from src import sentence_vector
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
        self.output = nn.Linear(int(conf.get("param","word_embedding_dim")), num_labels)
        # self.predict = nn.Linear(n_hidden, num_labels)

        self.double()
        # loss
        self.loss_function = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=float(conf.get("param", "lr_param")))
    def forward(self, input):
        out = self.output(input)
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
        # 计算准确率
        data_size = len(test_sentence_vectors)
        correct_num = 0

        for i in range(len(test_labels)):
            bow_vec = Variable(test_sentence_vectors[i])
            label = test_labels[i]
            output = self(bow_vec)

            pre_max_poss, index = torch.max(output, 1)

            if label == int(index):
                correct_num += 1

        return correct_num / data_size

if __name__ == '__main__':
    sentence_vectors,labels,test_sentence_vectors,test_labels = sentence_vector.bag_of_word_sentences(type='randomly')
    # label to idx
    output_size = len(set(labels))
    model = QuestionClassifier(output_size)

    for epoch in range(int(conf.get("param","epoch"))):
        model.train_model(sentence_vectors,labels)
        # 计算准确率
        acc = model.test_model(test_sentence_vectors,test_labels)
        print('epoch:', epoch, ' acc: ', acc)

