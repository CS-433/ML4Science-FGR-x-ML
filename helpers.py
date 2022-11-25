" Some helper function"

import torch 
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def get_single_dataset(path, features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','M_HI']):
    """
    Function to retrieve a single dataset.
    
    Args:
        path: path where to find the file
        features: list of features to extract from the dataset. M_HI must be always passed as last argument
    
    Returns:
        data: array of shape (N,len(features)-1)
        target: array of shape (N,) containing the true output value of each observation
     """
    f = h5py.File(path)

    target = f['M_HI'][:]

    data = np.empty((target.shape[0], len(features) -1) )

    for idx,feature in enumerate(features[:-1]):

        if feature == 'VelHalo':
            data[:,idx] = np.linalg.norm(f[feature][:])
        else:
            data[:,idx] = f[feature][:]

    return data, target, data.shape[1]


def min_max_scaling(x):
    """
    Function to compute minmax rescaling on a 1 dimensional numpy array.
    
    Args:
        x: array of shape (N,)
    
    Returns:
        x: array of shape (N,) containing the rescaled values
     """
    return (x-np.min(x))/(np.max(x)-np.min(x))


def visualization(losses_test, losses_train, R2_train, R2_test):
    """
    Visualization of loss of test (y axis) versus number of epoch that refer to that loss (x axis)

    Args:
        losses_test: array of shape (num_epochs,) containing test error
        losses_train: array of shape (num_epochs,) containing train error
    """
    fig,axs = plt.subplots(1,2, figsize=(15,10))
    axs[0].plot(range(1,len(losses_train)+1), losses_train, 'bo-', label='Loss(MSE) train')
    axs[0].plot(range(1,len(losses_test)+1), losses_test, 'ro-', label='Loss(MSE) test')
    axs[0].set(title='MSE w.r.t. number of epochs',xlabel='epochs',ylabel='test_loss(MSE)')
    axs[0].grid(visible=True)
    axs[0].set_yscale('log')
    axs[0].legend()

    axs[1].plot(range(1,len(R2_train)+1), R2_train, 'bo-', label='R2 score train')
    axs[1].plot(range(1,len(R2_test)+1), R2_test, 'ro-', label='R2 score test')
    axs[1].set(title='R2 score w.r.t. number of epochs',xlabel='epochs',ylabel='R2 score')
    axs[1].grid(visible=True)
    axs[1].legend()

def correlation_plot(predicted, y):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=4)
    ax.set_xlabel('Original')
    ax.set_ylabel('Predicted')

















