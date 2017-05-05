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

#Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--emb_path',type=str,help='Path to embedding (continue training)')
parser.add_argument('--netG', help="path to netG (to continue training)")

opt = parser.parse_args()
print(opt)

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
netG.load_state_dict(torch.load(opt.netG))
print(netG)


#Embedding Loading#
print('Load embeddings...')
class_embeddings = pickle.load(open(opt.emb_path,'rb'))
class_embeddings.cpu()
num_classes = class_embeddings.num_embeddings
emb_size = class_embeddings.embedding_dim
#class_embeddings.cuda()
print('Embedings dimension: %s | Number of classes: %s'%(emb_size,num_classes))
print('-'*89)
print()

#Generate noise and conditions to plot######
print('creating noise to plot')
noise_to_plot = torch.FloatTensor(num_classes*13, nz, 1, 1).normal_(0,1)
conditions_to_plot = np.arange(num_classes).repeat(13)
conditions_to_plot = torch.from_numpy(conditions_to_plot)
#noise_to_plot = noise_to_plot.cuda()
#conditions_to_plot = conditions_to_plot.cuda()
conditions_to_plot = Variable(conditions_to_plot)
noise_to_plot = Variable(noise_to_plot)

#printing
print('generating output')
fake = netG(noise_to_plot,conditions_to_plot,class_embeddings)
vutils.save_image(fake.data,
        'generated_samples.png',
        normalize=True,nrow=13)
print('done')