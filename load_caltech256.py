import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import shutil
import random
import tarfile
from torchvision.datasets.utils import download_url, list_dir, list_files
from torchvision.datasets import ImageFolder
caltech256_url_ = 'http://vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar'
download_url(caltech256_url_,'./data')

def extract_tar(filename):
    tar = tarfile.open(filename)
    tar.extractall('./data')
    tar.close()

def create_data(source_dir, target_dir, num_sample):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for files in os.listdir(source_dir):
        source_dir_0 =os.path.join(source_dir,files)
        target_dir_0 = os.path.join(target_dir,files)
        if not os.path.exists(target_dir_0):
            os.makedirs(target_dir_0)
        list_ = os.listdir(source_dir_0)
        numlist= random.sample(range(0,len(list_)),num_sample)
        for n in numlist:
            filename = list_[n]
            oldpath= os.path.join(source_dir_0,filename)
            shutil.move(oldpath,target_dir_0)

# extract downloaded tar file
if not os.path.exists('./data/256_ObjectCategories'):
    extract_tar('./data/256_ObjectCategories.tar')

# create train data
source = './data/256_ObjectCategories'
target = './data/caltech-256-60/train'
if not os.path.exists(target):
    create_data(source,target,60)
else:
    print('already split training set')
# create test data
source = './data/256_ObjectCategories'
target = './data/caltech-256-60/test'
if not os.path.exists(target):
    create_data(source,target,10)
else:
    print('already split testing set')

# load training data and testing data
def load_data(target_train=None, target_test=None):
    target_train = './data/caltech-256-60/train'
    target_test = './data/caltech-256-60/test'
    preprocess_input = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = ImageFolder(target_train, preprocess_input)
    test_dataset = ImageFolder(target_test, preprocess_input)
    num_class = os.listdir(target_test).__len__()
    return train_dataset, test_dataset,num_class

#train_data, test_data, num_class = load_data()
#print(num_class)
