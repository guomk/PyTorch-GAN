import argparse
import sys
import os

from models import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--saved_model", type=str, default="apple2orange", help="name of the training dataset")
parser.add_argument("--dataset_name", type=str, default="apple2orange", help="name of the test dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")

opt = parser.parse_args()
print(opt)

import numpy as np
import glob

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import Generator
from datasets import ImageDataset


cuda = True if torch.cuda.is_available() else False

###### Definition of variables ######

# Dimensionality (channel-wise) of image embedding
shared_dim = opt.dim * 2 ** opt.n_downsample

# Networks
shared_E = ResidualBlock(features=shared_dim)
E1 = Encoder(in_channels=opt.channels, dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
E2 = Encoder(in_channels=opt.channels, dim=opt.dim, n_downsample=opt.n_downsample, shared_block=shared_E)
shared_G = ResidualBlock(features=shared_dim)
netG_A2B = Generator(out_channels=opt.channels, dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
netG_B2A = Generator(out_channels=opt.channels, dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)

if cuda:
    E1.cuda()
    E2.cuda()
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
if cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# ANCHOR For now we load the checkpoint from last epoch
E1.load_state_dict(torch.load("saved_models/%s/E1_%d.pth" % (opt.saved_model, opt.n_epochs-1), map_location='cuda:0'))
E2.load_state_dict(torch.load("saved_models/%s/E2_%d.pth" % (opt.saved_model, opt.n_epochs-1), map_location='cuda:0'))
netG_A2B.load_state_dict(torch.load("saved_models/%s/G1_%d.pth" % (opt.saved_model, opt.n_epochs-1), map_location='cuda:0'))
netG_B2A.load_state_dict(torch.load("saved_models/%s/G2_%d.pth" % (opt.saved_model, opt.n_epochs-1), map_location='cuda:0'))

E1.to(device)
E2.to(device)
netG_A2B.to(device)
netG_B2A.to(device)

# Set model's test mode
E1.eval()
E2.eval()
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Dataset loader
if opt.channels == '3':
    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
else:
    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) ]
dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)
###################################

###### Testing######

# Create output dirs if they don't exist
dataset_name = opt.dataset_name
save_dir = "inference/" + dataset_name + "/"
if not os.path.exists(save_dir + 'A'):
    os.makedirs(save_dir + 'A')
if not os.path.exists(save_dir + 'B'):
    os.makedirs(save_dir + 'B')

# Prepare file names
filename_A, filename_B = [], []
dataroot = "../../data/%s" % opt.dataset_name
for file in os.listdir(dataroot + "/test/A"):
    if file.endswith(".jpg") or file.endswith(".png"):
        filename_A.append(file[:-4]) # Strip off file format (.jpg)

for file in os.listdir(dataroot + "/test/B"):
    if file.endswith(".jpg") or file.endswith(".png"):
        filename_B.append(file[:-4]) # Strip off file format (.jpg)



for i, batch in enumerate(dataloader):
    # Skip the final batch when the total number of training images modulo batch-size does not equal zero
    if len(batch['A']) != opt.batch_size or len(batch['B']) != opt.batch_size:
        print("Batch Mismatch - skip current batch")
        continue   # ANCHOR

    

    # Set model input
    real_A = Variable(batch["A"].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor))

    # Generate output
    _, Z1 = E1(real_A)
    _, Z2 = E2(real_B)
    fake_A = netG_B2A(Z2)
    fake_B = netG_A2B(Z1)

    # Save image files
    save_image(fake_A, save_dir + f'A/{filename_A[i]}_fake.png')
    save_image(fake_B, save_dir + f'B/{filename_A[i]}_fake.png')

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
