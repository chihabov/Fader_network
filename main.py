from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
import torch
from torch import nn
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset__1 import DatasetCelebA
#from utils import create_dataloader
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from natsort import natsorted , ns
path ='/home/master/Desktop/prfojetMLA/dataCeleb'
import os

data=os.listdir(path)
data=natsorted(data,alg=ns.PATH | ns.IGNORECASE)
print(data[0])
path ='/home/master/Desktop/prfojetMLA/dataCeleb'

dataset = DatasetCelebA(root=path,
                        attr='/home/master/Desktop/prfojetMLA/list_attr_celeba.csv')
#np.random.seed(42)
subset_indices = np.random.choice(len(dataset), 100, replace=False)
dataloader = DataLoader(dataset, batch_size=42,
                        sampler=SubsetRandomSampler(subset_indices),
                        num_workers = 4)





