import numpy as np
import torch
import os
import random

from configparser import ConfigParser
# use conf to read the configuration
conf = ConfigParser()
conf.read('../data/bow.config')
print(conf.get('param','model'))
def splitData():
    path = os.path.join(os.getcwd(),"..","data", "train_5500.txt")
    f = open(path)
    lines = f.readlines()
    f.close()
    train, dev = split(lines, shuffle=True, ratio=0.9)

    # read
    file = open(conf.get('param','path_train'), 'w')
    for i in range(len(train)):
        file.write(train[i])
    file.close()

    file = open(conf.get('param','path_dev'), 'w')
    for i in range(len(dev)):
        file.write(dev[i])
    file.close()

def split(full_list, shuffle=False, ratio=0.2):
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