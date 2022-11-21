" Some helper function"

import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset
import matplotlib as plt

def get_single_dataset(path, features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','M_HI']):
    
    " Return data from file in path. M_HI must be passed as last argument"

    f = h5py.File(path)

    target = f['M_HI'][:]

    data = np.empty((target.shape[0], len(features) -1) )

    for idx,feature in enumerate(features[:-1]):

        if feature == 'VelHalo':
            data[:,idx] = np.linalg.norm(f[feature][:])
        else:
            data[:,idx] = f[feature][:]

    return data,target, len(features)-1


def min_max_scaling(x):
    return (x-np.min(x))/(np.max(x)-np-min(x))


def visualize_LvsN(losses_test, losses_train):

    "Visualization of loss of test (y axis) versus number of epoch that refer to that loss (x axis)"

    plt.figure(figsize=(7,9))
    plt.plot(range(1,len(losses_test)+1), losses_test, 'ro-', label='Loss(MSE) test')
    plt.plot(range(1,len(losses_train)+1), losses_train, 'ro-', label='Loss(MSE) test')
    plt.title('MSE in function of number of epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss(MSE)')
    plt.legend()

















