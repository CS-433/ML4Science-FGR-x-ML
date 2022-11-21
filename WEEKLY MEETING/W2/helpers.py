" Some helper function"

import torch
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset

def get_single_dataset(path, features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','M_HI']):
    
    " Return data from file in path. M_HI must be passed as last argument"

    f = h5py.File(path)

    target = f['M_HI']

    data = np.empty(target.shape[0], len(features) -1 )

    for idx,feature in enumerate(features[:-1]):

        if feature == 'VelHalo':
            data[:,idx] = np.linalg.norm(f[feature])
        else:
            data[:,idx] = f[feature]

    return data,target, len(features)-1

def visualize_loss():
    pass















