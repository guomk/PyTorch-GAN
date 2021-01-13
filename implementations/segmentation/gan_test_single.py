import argparse
import sys
import os

from models import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--saved_model", type=str, default="applG_BAorange", help="name of the training dataset")
parser.add_argument("--dataset_name", type=str, default="applG_BAorange", help="name of the test dataset")
parser.add_argument("--surfix", type=str, default="", help="surfix of saving directories")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of preferred image height")
parser.add_argument("--img_width", type=int, default=256, help="size of prefereed image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--gpu", type=str, default='0,1', help='choose which gpu(s) to use during training')
parser.add_argument("--domain", type=str, default='B', help='domain of the input image (A/B)')


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
import torch.nn.functional as F

from models import *
from datasets import *


cuda = True if torch.cuda.is_available() else False

mode = 'test' # train or test

###### Definition of variables ######

input_shape = (opt.channels, opt.img_height, opt.img_width)


# Networks
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)

if cuda:
    G_AB.cuda()
    G_BA.cuda()

# Load state dicts
if cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# ANCHOR For now we load the checkpoint from last epoch
G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.saved_model + opt.surfix, opt.n_epochs-1), map_location='cuda:0'))
G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.saved_model + opt.surfix, opt.n_epochs-1), map_location='cuda:0'))

G_AB.to(device)
G_BA.to(device)

# Set model's test mode
G_AB.eval()
G_BA.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Dataset loader
if opt.channels == '3':
    transforms_ = [ 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
else:
    transforms_ = [ 
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)) ]


dataloader = DataLoader(
    OneDomainImageDataset("../../../data/%s" % opt.dataset_name, transforms_=transforms_, mode=mode),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

# get shape of the incoming data
input_dim = [0,0,0,0]
for i, batch in enumerate(dataloader):
    input_dim = batch[opt.domain].shape
    break

width, height = input_dim[3], input_dim[2]
print(input_dim)

# if opt.img_width > width or opt.img_height > height:
#     # TODO Fix padding logic
#     horizontal_pad = (opt.img_width - width) // 2
#     vertical_pad = (opt.img_height - height) // 2

#     # print(horizontal_pad)
#     # print(vertical_pad)

#     if opt.channels == '3':
#         transforms_ = [ transforms.ToTensor(),
#                         transforms.Pad([horizontal_pad, vertical_pad], padding_mode='edge'),
#                         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
#     else:
#         transforms_ = [ transforms.ToTensor(),
#                         transforms.Pad([horizontal_pad, vertical_pad], padding_mode='edge'),
#                         transforms.Normalize((0.5,), (0.5,)) ]

#     dataloader = DataLoader(
#         ImageDataset("../../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=False, mode="test"),
#         batch_size=1,
#         shuffle=False,
#         num_workers=1,
#     )


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
dataroot = "../../../data/%s" % opt.dataset_name
if opt.domain == 'A':
    for file in os.listdir(dataroot + f"/{mode}/imgs"):
        if file.endswith(".jpg") or file.endswith(".png"):
            filename_A.append(file[:-4]) # Strip off file format (.jpg)
elif opt.domain == 'B':
    for file in os.listdir(dataroot + f"/{mode}/imgs"):
        if file.endswith(".jpg") or file.endswith(".png"):
            filename_B.append(file[:-4]) # Strip off file format (.jpg)
else:
    print("Domain option not supported")
    exit 



for i, batch in enumerate(dataloader):

    # Set model input
    if opt.domain == 'A':
        real_A = Variable(batch["A"].type(Tensor))
        fake_B = G_AB(real_A)

        img_sample = torch.cat((real_A.data, fake_B.data), 0)
        save_image(img_sample, save_dir + f'{filename_A[i]}.png', nrow=1, padding=0, normalize=True)
        save_image(real_A, save_dir + f'A/{filename_A[i]}.png', nrow=1, normalize=True)
        save_image(fake_B, save_dir + f'B/{filename_A[i]}_fake.png', nrow=1, normalize=True)
    elif opt.domain == 'B':
        real_B = Variable(batch["B"].type(Tensor))
        # print(real_B.shape)
        fake_A = G_BA(real_B)
        # print(fake_A.shape)

        pad  = torch.nn.ReflectionPad2d([1,1,1,1])
        fake_A = pad(fake_A)

        # print(fake_A.shape)

        img_sample = torch.cat((real_B.data, fake_A.data), 0)   
        save_image(img_sample, save_dir + f'{filename_B[i]}.png', nrow=1, padding=0, normalize=True)
        save_image(fake_A, save_dir + f'A/{filename_B[i]}.png', nrow=1, normalize=True)
        save_image(real_B, save_dir + f'B/{filename_B[i]}.png', nrow=1, normalize=True)

    # Save image files
    # img_sample = torch.cat((real_A.data, fake_B.data, real_B.data, fake_A.data), 0)
    # print(fake_A - real_B)
    # print(real_A.shape)
    # print(real_B.shape)
    # print(fake_A.shape)
    # print(fake_B.shape)
    # print(img_sample.shape)
    # break
    # save_image(img_sample, save_dir + f'{filename_A[i]}.png', nrow=1, padding=0, normalize=True)
    # save_image(fake_A, save_dir + f'A/{filename_A[i]}_fake.png', nrow=1, normalize=True)
    # save_image(fake_B, save_dir + f'B/{filename_B[i]}_fake.png', nrow=1, normalize=True)

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
