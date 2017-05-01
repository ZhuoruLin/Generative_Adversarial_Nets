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
import numpy as np
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
##Simon's Edit
parser.add_argument('--emb_size',type=int, default=64,help='embedding size for condition label')
parser.add_argument('--nc',type=int,default=3,help='number of channel of image. Default 1. Change if using MNIST')
parser.add_argument('--printper',type=int,default=100,help='Number of steps each print')
parser.add_argument('--emb_path',type=str,default='',help='Path to embedding (continue training)')
parser.add_argument('--saveper',type=int,default=1,help='Number of steps each save')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    #Simon's Edit add support of conditional GAN
    if os.path.exists(os.path.join(opt.dataroot,'.DS_Store')):
        dataset.train_labels = [x[1]-1 for x in dataset.imgs]
    else:
        dataset.train_labels = [x[1] for x in dataset.imgs]
        
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
##########Update MNIST###############
elif opt.dataset == 'MNIST':
    dataset = dset.MNIST(opt.dataroot,train=True, download=True,
                   transform=transforms.Compose([
                transforms.Scale(opt.imageSize),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
#############################
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
if opt.dataset == 'MNIST':
    nc = 1


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.inserted_dim = nz+opt.emb_size
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
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, catted_input, range(self.ngpu))
        else:
            output = self.main(catted_input)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),#original
            nn.Conv2d(ndf*8,opt.emb_size,4,1,0,bias=False),
            # Simon's Edit
            #nn.Sigmoid()
        )
        self.decode = nn.Sequential(
            nn.Linear(2*opt.emb_size,opt.emb_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(opt.emb_size,1),
            nn.Sigmoid()
        )

    def forward(self, input,condition=None,embedding_dict=None):
        #Get embedding
        emb = embedding_dict(condition)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            intermediate = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            intermediate = self.main(input)
        num_samples=condition.size()[0]
        to_cat = intermediate.view(num_samples,-1)
        catted = torch.cat([to_cat,emb],1)
        output = self.decode(catted)

        return output.view(-1, 1)

netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

##################Embeddings for class label#########
num_classes = len(np.unique(dataset.train_labels))
if isinstance(dataset.train_labels,torch.LongTensor):
    num_classes = len(np.unique(dataset.train_labels.numpy()))
class_embeddings = nn.Embedding(embedding_dim=opt.emb_size,num_embeddings=num_classes)
if opt.emb_path != '':
    class_embeddings = pickle.load(open(opt.emb_path,'rb'))
condition = torch.LongTensor(opt.batchSize)
#####################################################

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    #fix gpu computing for embedding
    class_embeddings = class_embeddings.cuda()
    condition = condition.cuda()

input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
##############
condition = Variable(condition)

###########################
#create 10 plots for each condition
noise_to_plot = torch.FloatTensor(num_classes*8, nz, 1, 1).normal_(0,1)
conditions_to_plot = np.arange(num_classes).repeat(8)
conditions_to_plot = torch.from_numpy(conditions_to_plot)
if opt.cuda:
    noise_to_plot = noise_to_plot.cuda()
    conditions_to_plot = conditions_to_plot.cuda()
conditions_to_plot = Variable(conditions_to_plot)
noise_to_plot = Variable(noise_to_plot)




# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))




for epoch in range(opt.niter):
    for i, (data,image_label) in enumerate(dataloader):
        condition.data.resize_(image_label.size()).copy_(image_label)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label)

        output = netD(input,condition,class_embeddings)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        fake = netG(noise,condition,class_embeddings)
        label.data.fill_(fake_label)
        output = netD(fake.detach(),condition,class_embeddings)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake,condition,class_embeddings)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

        
        ##########################
        if i % opt.printper == 0:
#             vutils.save_image(real_cpu,
#                     '%s/real_samples.png' % opt.outf,
#                     normalize=True)
            fake = netG(noise_to_plot,conditions_to_plot,class_embeddings)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    if epoch % opt.saveper ==0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
        with open('%s/embDict_epoch_%d.pth'%(opt.outf,epoch),'wb') as f:
            if opt.cuda:
                pickle.dump(file=f,obj=class_embeddings.cpu())
            else:
                pickle.dump(file=f,obj=class_embeddings)
