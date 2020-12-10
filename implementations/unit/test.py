import argparse
import sys
import os

from models import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--saved_model", type=str, default="apple2orange", help="name of the training dataset")
parser.add_argument("--dataset_name", type=str, default="apple2orange", help="name of the test dataset")
parser.add_argument("--surfix", type=str, default="", help="surfix of saving directories")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--gpu", type=str, default='0,1', help='choose which gpu(s) to use during training')

opt = parser.parse_args()

if opt.surfix != "":
    opt.surfix = '_' + opt.surfix

print(opt)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu


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
G1 = Generator(out_channels=opt.channels, dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)
G2 = Generator(out_channels=opt.channels, dim=opt.dim, n_upsample=opt.n_downsample, shared_block=shared_G)

if cuda:
    E1.cuda()
    E2.cuda()
    G1.cuda()
    G2.cuda()

# Load state dicts
if cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# ANCHOR For now we load the checkpoint from last epoch
E1.load_state_dict(torch.load("saved_models/%s/E1_%d.pth" % (opt.saved_model + opt.surfix, opt.n_epochs-1), map_location='cuda:0'))
E2.load_state_dict(torch.load("saved_models/%s/E2_%d.pth" % (opt.saved_model + opt.surfix, opt.n_epochs-1), map_location='cuda:0'))
G1.load_state_dict(torch.load("saved_models/%s/G1_%d.pth" % (opt.saved_model + opt.surfix, opt.n_epochs-1), map_location='cuda:0'))
G2.load_state_dict(torch.load("saved_models/%s/G2_%d.pth" % (opt.saved_model + opt.surfix, opt.n_epochs-1), map_location='cuda:0'))

E1.to(device)
E2.to(device)
G1.to(device)
G2.to(device)

# Set model's test mode
E1.eval()
E2.eval()
G1.eval()
G2.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Dataset loader
if opt.channels == '3':
    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
else:
    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) ]

# if opt.channels == 3:
#     transforms_ = [
#         transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
#         transforms.RandomCrop((opt.img_height, opt.img_width)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ]
# else:
#     transforms_ = [
#         transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
#         transforms.RandomCrop((opt.img_height, opt.img_width)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5), (0.5)),
#     ]

dataloader = DataLoader(
    ImageDataset("../../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=1,
)
###################################

###### Testing######

# Create output dirs if they don't exist
dataset_name = opt.dataset_name
save_dir = "inference/" + dataset_name + opt.surfix + "/"
if not os.path.exists(save_dir + 'A'):
    os.makedirs(save_dir + 'A')
if not os.path.exists(save_dir + 'B'):
    os.makedirs(save_dir + 'B')

# Prepare file names
filename_A, filename_B = [], []
dataroot = "../../../data/%s" % opt.dataset_name + opt.surfix
for file in os.listdir(dataroot + "/test/A"):
    if file.endswith(".jpg") or file.endswith(".png"):
        filename_A.append(file[:-4]) # Strip off file format (.jpg)

for file in os.listdir(dataroot + "/test/B"):
    if file.endswith(".jpg") or file.endswith(".png"):
        filename_B.append(file[:-4]) # Strip off file format (.jpg)



for i, batch in enumerate(dataloader):
    # Skip the final batch when the total number of training images modulo batch-size does not equal zero
    # if len(batch['A']) != opt.batch_size or len(batch['B']) != opt.batch_size:
    #     print("Batch Mismatch - skip current batch")
    #     continue   # ANCHOR


    # Set model input
    real_A = Variable(batch["A"].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor))

    # Generate output
    _, Z1 = E1(real_A)
    _, Z2 = E2(real_B)
    fake_A = G1(Z2)
    fake_B = G2(Z1)

    # Save image files
    # img_sample = torch.cat((real_A.data, fake_B.data, real_B.data, fake_A.data), 0)
    # save_image(img_sample, save_dir + f'{filename_A[i]}.png', nrow=1, normalize=True)
    save_image(fake_A, save_dir + f'A/{filename_A[i]}_fake.png', nrow=1, normalize=True)
    save_image(fake_B, save_dir + f'B/{filename_B[i]}_fake.png', nrow=1, normalize=True)

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
