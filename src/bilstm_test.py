import numpy as np
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from src.preprocessing import process_train_set,process_new_dataset
from src import word2vec as w2v
from src import classifier
from src.global_value import conf
import src.question_classifier

torch.manual_seed(1)
random.seed(1)


class BiLSTMTagger(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        """
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        """
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                           bidirectional=True, dropout=0.15)
        """
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                   bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 50)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        #_pad, _len = rn.pad_packed_sequence(x, batch_first=True)
        embedding = self.dropout(self.embedding(x))
        #embedding = self.embedding(x)
        embedding =  embedding.permute(1,0,2)
        output, (hidden, cell) = self.rnn(embedding)

        # [num_layers*2, b, hid_dim] => 2 of [b, hid_dim] => [b, hid_dim*2]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)


        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        '''
        output = output[-1]
        output = self.dropout(output)
        out = self.fc(output)'''

        return out,cell


def acc_(preds, y):
    """
    get accuracy
    """
    preds = preds.max(dim=1, keepdim=True)[1].squeeze(1)

    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    #print(acc)
    return acc


def padding_feature(int_word,seq_length):
   
    '''
    :param sentence2idx:
    :return: feature of sentences, if the length of sentences is less than seq_length,padding with 0
    else truncated to the seq_length
    '''
    feature  = np.zeros((len(int_word),seq_length),dtype = int)
    for i,row in enumerate(int_word):
        feature[i,-len(row):] = np.array(row)[:seq_length]

    assert len(feature) == len(int_word), "Your features should have as many rows as reviews."
    assert len(feature[0]) == seq_length, "Each feature row should contain seq_length values."
    return feature


def to_dataloader(feature,labels):
    labels = np.array(labels)
    train_data = TensorDataset(torch.from_numpy(feature), torch.from_numpy(labels))
    # dataloaders
    batch_size = int(conf.get('param','batch_size'))

    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    return train_loader


def train_another_new(rnn, train_loader, optimizer, criteon):
    avg_acc = []
    #rnn.train()
    cells = []

    for inputs, labels in train_loader:
        '''
        single_label = []
        inputs = torch.Tensor(inputs).unsqueeze(-1).long()
        single_label.append(labels)
        labels = torch.Tensor(single_label).long()'''
        # pred = rnn(batch.text).max(dim=1, keepdim=True)[1].squeeze(1)
        pred,cell = rnn(inputs.long())
        # pred = rnn(inputs).squeeze(1)
        pred = pred.squeeze(1)
        loss = criteon(pred, labels.long())
        # acc = binary_acc(pred, batch.label).item()
        # avg_acc.append(acc)
        acc = acc_(pred, labels.long()).item()
        '''
        single_label = []
        inputs = torch.Tensor(inputs).unsqueeze(-1).long()
        single_label.append(labels)
        labels = torch.Tensor(single_label).long()
        # pred = rnn(batch.text).max(dim=1, keepdim=True)[1].squeeze(1)
        pred, cell = rnn(inputs)
        # pred = rnn(inputs).squeeze(1)
        pred = pred.squeeze(1)
        loss = criteon(pred, labels)
        # acc = binary_acc(pred, batch.label).item()
        # avg_acc.append(acc)
        acc = acc_(pred, labels).item()'''

        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cells.append(cell)
    return cells

    # print(acc)


def eval(rnn, test_loader, criteon, patience=50):
        avg_acc = []
        rnn.eval()
        with torch.no_grad():
            curr_patience = 0
            min_loss = 10e9
            for inputs, labels in test_loader:
                pred, cell = rnn(inputs.long())
                pred = pred.squeeze(-1)
                loss = criteon(pred, labels.long())
                acc = acc_(pred, labels.long()).item()
                avg_acc.append(acc)
                # for early stopping
                if loss < min_loss:
                    curr_patience = 0
                    min_loss = loss
                else:
                    curr_patience += 1
                if curr_patience == patience:
                    print('Patience condition met, do early stopping!\n')
                    break

        avg_acc = np.array(avg_acc).mean()
        print('test acc:', avg_acc)


def refactor(sen, labels):
    data = []
    for i in range(0, len(labels)):
        data.append((sen[i], labels[i]))
    return data


'''
with open('./a.txt',"w") as f:
        for batch in range(len(cells)):
            cel = cells[batch].permute(1,0,2)
            cel = cel.reshape(len(cel),-1)
            for i in range(len(cel)):
                Q = cel[i].data.numpy()  # tensor转换成array
                np.savetxt(f, Q[None], delimiter=' ', newline='\n')
'''

def train_Bilstm():
    train_int_word, train_int_label, word2idx, lable2idx = process_train_set(conf.get('param', 'path_train'))
    test_int_word, test_int_label = process_new_dataset(word2idx, lable2idx, conf.get('param', 'path_dev'))

    seq_length = 10
    feature_metrix = padding_feature(train_int_word, seq_length)
    feature_metrix_dev = padding_feature(test_int_word, seq_length)

    train_loader = to_dataloader(feature_metrix, train_int_label)
    test_loader = to_dataloader(feature_metrix_dev, test_int_label)


    rnn_ = BiLSTMTagger(len(word2idx), int(conf.get("param", "word_embedding_dim")), 100)

    if conf.get('param','pre_train'):
        word2vec = w2v.read_word2vec(conf.get("param","path_pre_emb"))
        vocab = torch.tensor(word2vec)
        pretrained_embedding = vocab
        print('pretrained_embedding:', pretrained_embedding.shape)
        rnn_.embedding.from_pretrained(pretrained_embedding,freeze=conf.get('param','freeze'))

    device = 'cpu'
    optimizer = optim.Adam(rnn_.parameters(), lr=float(conf.get("param","lr_param")))
    criteon = nn.CrossEntropyLoss().to(device)
    rnn_.to(device)
    print(rnn_.to(device))

    for epoch in range(int(conf.get("param","epoch"))):
        cells = train_another_new(rnn_, train_loader, optimizer, criteon)
        eval(rnn_, test_loader, criteon, patience=int(conf.get('param', 'early_stopping')))
    torch.save(rnn_, 'word2vec_Bilstm_3.pkl')
    #print('```````````````')
    classifier.vector_file()
    

if __name__ == '__main__':
    train_Bilstm()
