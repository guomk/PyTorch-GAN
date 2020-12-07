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
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--style_dim", type=int, default=8, help="dimensionality of the style code")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")

opt = parser.parse_args()
print(opt)

import numpy as np
import glob

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models import *
from datasets import *


cuda = True if torch.cuda.is_available() else False

# Networks
Enc1 = Encoder(in_channels=opt.channels, dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec1 = Decoder(out_channels=opt.channels, dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Enc2 = Encoder(in_channels=opt.channels, dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec2 = Decoder(out_channels=opt.channels, dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)

if cuda:
    Enc1.cuda()
    Enc2.cuda()
    Dec1.cuda()
    Dec2.cuda()

# Load state dicts
if cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# ANCHOR For now we load the checkpoint from last epoch
Enc1.load_state_dict(torch.load("saved_models/%s/Enc1_%d.pth" % (opt.saved_model, opt.n_epochs-1), map_location='cuda:0'))
Enc2.load_state_dict(torch.load("saved_models/%s/Enc2_%d.pth" % (opt.saved_model, opt.n_epochs-1), map_location='cuda:0'))
Dec1.load_state_dict(torch.load("saved_models/%s/Dec1_%d.pth" % (opt.saved_model, opt.n_epochs-1), map_location='cuda:0'))
Dec2.load_state_dict(torch.load("saved_models/%s/Dec2_%d.pth" % (opt.saved_model, opt.n_epochs-1), map_location='cuda:0'))

Enc1.to(device)
Enc2.to(device)
Dec1.to(device)
Dec2.to(device)

# Set model's test mode
Enc1.eval()
Enc2.eval()
Dec1.eval()
Dec2.eval()

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
    CycleGANImageDataset("../../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
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
dataroot = "../../../data/%s" % opt.dataset_name
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
    X1 = Variable(batch["A"].type(Tensor)) # real_A
    X2 = Variable(batch["B"].type(Tensor)) # real_B

    # Sampled style codes
    style_1 = Variable(torch.randn(X1.size(0), opt.style_dim, 1, 1).type(Tensor))
    style_2 = Variable(torch.randn(X2.size(0), opt.style_dim, 1, 1).type(Tensor)) # ANCHOR modified from X1 to X2

    # Get shared latent representation
    c_code_1, s_code_1 = Enc1(X1)
    c_code_2, s_code_2 = Enc2(X2)

    # Reconstruct images
    # X11 = Dec1(c_code_1, s_code_1)
    # X22 = Dec2(c_code_2, s_code_2)

    # Translate images
    X21 = Dec1(c_code_2, style_1) # fake_A
    X12 = Dec2(c_code_1, style_2) # fake_B

    # Save image files
    save_image(X21, save_dir + f'A/{filename_A[i]}_fake.png')
    save_image(X12, save_dir + f'B/{filename_B[i]}_fake.png')

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################