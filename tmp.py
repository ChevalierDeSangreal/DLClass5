import os
import torch
from torch.nn import functional as F
from torch import nn
from DataLoader import load_data
from utils import try_gpu, Accumulator, Timer, accuracy, evaluate_accuracy_gpu

batch_size = 512
train_iter = load_data(batch_size)

for i, (X, y) in enumerate(train_iter):
    print('end')