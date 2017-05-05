from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import pickle
import numpy as np

f=open('embDict.pth','rb')
emb = pickle.load(f)
emb.cpu()
save_file = open('embDict_cpu.pth','wb')
pickle.dump(emb,save_file)