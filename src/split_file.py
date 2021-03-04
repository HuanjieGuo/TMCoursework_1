from src.question_classifier import conf
import random
import os
'''
descript:
Run this function and it will split the train_5500.txt
into dev.txt and train.txt
'''
def get_train_dev():
    # open the data file
    path = os.path.join(os.getcwd(),"..","data", "train_5500.txt")
    f = open(path)
    lines = f.readlines()
    f.close()

    # split it into 1:9
    train, dev = random_split(lines, shuffle=True, ratio=0.9)

    # write the train data into the train.txt
    file = open(conf.get('param','path_train'), 'w')
    for i in range(len(train)):
        file.write(train[i])
    file.close()

    # write the dev data into the train.txt
    file = open(conf.get('param','path_dev'), 'w')
    for i in range(len(dev)):
        file.write(dev[i])
    file.close()

'''
param:
full_list: total data
ratio: 0-1, the ratio of the first set.
'''
def random_split(full_list, shuffle=False, ratio=0):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

if __name__ == '__main__':
    pass