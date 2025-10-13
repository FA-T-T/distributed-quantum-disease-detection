import torch
import torch.nn as nn
import argparse
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch.utils.data as dataset
from torch.nn.functional import one_hot
import torch.utils.data.dataloader as dataloader
import time
import matplotlib.pyplot as plt
import tqdm
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3*128*128,1024),
            nn.Sigmoid(),
            # nn.Linear(1024,524),
            # nn.Dropout(0.35),
            nn.Linear(1024,6),
            # nn.Softmax()
        )
    
    def forward(self, x):
        x=x.view(x.size(0),-1)
        x=self.mlp(x)
        return x
    

