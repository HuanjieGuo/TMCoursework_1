'''
This file will be directly run by TA
it captures the option and the parameters input by the user.
you can run it on console by:
python3 question_classifier.py  --test --config "../data/bow.config"
'''

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))

rootPath = os.path.split(curPath)[0]

sys.path.append(rootPath)


import torch
from src import global_value as gv
import argparse
from src import bow_nn

torch.manual_seed(1)

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--config', help='The path of the configuration file',type=str, default='../data/bow.config')
parser.add_argument("--train", help="To train the model",
                    action="store_true")
parser.add_argument("--test", help="To test the model",
                    action="store_true")

args = parser.parse_args()

# use conf to read the configuration
gv.conf.read(args.config)

if(args.train):
    # do the train function
    if(gv.conf.get("param","model")=="bow"):
        bow_nn.train()


if(args.test):
    # do the test function
    if(gv.conf.get("param","model")=="bow"):
        bow_nn.test()

if __name__ == '__main__':
    # print(conf.get('param', 'model'))
    pass