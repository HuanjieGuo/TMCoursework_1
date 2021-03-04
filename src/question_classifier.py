'''
This file will be directly run by TA
it captures the option and the parameters input by the user.
you can run it on console by:
python3 question_classifier.py  --test --config "../data/bow.config"
'''

import numpy as np
import torch
import random
from configparser import ConfigParser
import argparse

torch.manual_seed(1)
random.seed(1)

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--config', help='The path of the configuration file',type=str, default='../data/bow.config')
parser.add_argument("--train", help="To train the model",
                    action="store_true")
parser.add_argument("--test", help="To test the model",
                    action="store_true")

args = parser.parse_args()

# use conf to read the configuration
conf = ConfigParser()
conf.read(args.config)

if(args.train):
    # do the train function
    print('train')

if(args.test):
    # do the test function
    print('test')

if __name__ == '__main__':
    # print(conf.get('param', 'model'))
    pass