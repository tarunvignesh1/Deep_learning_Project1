import torch


#Checking for availablilty of CUDA
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
# importing essential Libraries


import os
import numpy as np
import torchvision
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import splitfolders


datapath = 'D:\TarunDocs\Kaggle\data'


splitfolders.ratio(datapath, output="D:\TarunDocs\Kaggle\data\splited", seed=1337, ratio=(.8, .1, .1), group_prefix=None)


Newpath = 'D:/TarunDocs/Kaggle/data/splited'

trainpath = os.path.join(Newpath,'train/')
valpath = os.path(Newpath,'val/')
testpath = os.path(Newpath,'test/')








