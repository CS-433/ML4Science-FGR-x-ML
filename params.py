" Useful variables to define model"

import torch

first_run= True
name_file = 'LH_0/MHI_LH0_z=0.770.hdf5'
path_file = './outputs_test/'+ name_file
batch_size = 512
lr = 0.1
epochs = 30
dtype = torch.float64
activation = 'sigmoid' 


