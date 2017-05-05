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

#Necessary model definition

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ngf=128
nz=50
emb_size=50
nc=3
  
class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.inserted_dim = nz+emb_size
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     self.inserted_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input,condition = None,embedding_dict = None):
        emb = embedding_dict(condition)
        catted_input = torch.cat([input,emb],1)
        output = self.main(catted_input)
        #if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #    output = nn.parallel.data_parallel(self.main, catted_input, range(self.ngpu))
        #else:
        #    output = self.main(catted_input)
        return output

ngpu=0
nz=50
netG = _netG(ngpu)
netG.apply(weights_init)
#Loading netG
print('Loading netG')
netG.load_state_dict(torch.load('netG.pth'))
netG.cpu()
torch.save(netG.state_dict(),'netG_cpu.pth')
print(netG)
