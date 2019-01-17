import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import tensor
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import custom_utils as utils

import os
import json
import sys
import argparse

def main():
    in_arg = get_input_args()
    
    model, optimizer, criterion = utils.load_checkpoint(in_arg.checkpoint)
    probs, classes = utils.predict(in_arg.image_filepath, model, in_arg.top_k)
    class_values = []
    
    #cat_to_name
    if in_arg.category_names != None:
        print('using custom name mapping')
        cat_to_name = load_category_name_mappings(in_arg.category_names)
    else:
        print('using default name mapping')
        cat_to_name = load_category_name_mappings('cat_to_name.json')

    for c, prob  in zip (classes, probs[0]):
        print("{}: {}".format(cat_to_name[c], prob))

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_filepath', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--category_names', type=str, default=None)
    parser.add_argument('--gpu', default=False, type=lambda x: (str(x).lower() == 'true'))
    
    return parser.parse_args()

def load_category_name_mappings(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name
    

# Call to main function to run the program
if __name__ == "__main__":
    main()