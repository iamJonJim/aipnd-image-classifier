#import matplotlib.pyplot as plt

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
  
    dataloaders, image_datasets = utils.load_data(in_arg.data_dir)
    model, optimizer, criterion = utils.nn_config(in_arg.dropout, in_arg.hidden_size, in_arg.learning_rate, in_arg.arch, in_arg.gpu)
    utils.nn_train(model, optimizer, criterion, in_arg.epochs, dataloaders, in_arg.gpu)
    utils.save_trained_nn(in_arg.save_dir, model, optimizer, in_arg.arch, in_arg.hidden_size, in_arg.learning_rate, in_arg.dropout, in_arg.epochs, image_datasets)

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default="./flowers")
    parser.add_argument('--arch', type=str, default="vgg")
    parser.add_argument('--save_dir', type=str, default="./train_checkpoint.pth")
    parser.add_argument('--hidden_size', type=int, dest="hidden_size",default=120)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--gpu', default=False, type=lambda x: (str(x).lower() == 'true'))
    
    return parser.parse_args()
    

# Call to main function to run the program
if __name__ == "__main__":
    main()