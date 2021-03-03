import numpy as np
import torch
from configparser import ConfigParser

# use conf to read the configuration
conf = ConfigParser()
conf.read('../data/bow.config')

if __name__ == '__main__':
    print(conf.get('param', 'model'))