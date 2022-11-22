" Some helper function"

import torch 
import torch.nn as nn
import numpy as np
import h5py
from torch.utils.data import Dataset
import matplotlib as plt


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


def visualize_LvsN(losses_test, losses_train):
    """
    Visualization of loss of test (y axis) versus number of epoch that refer to that loss (x axis)

    Args:
        losses_test: array of shape (num_epochs,) containing test error
        losses_train: array of shape (num_epochs,) containing train error
    """
    plt.figure(figsize=(7,9))
    plt.plot(range(1,len(losses_test)+1), losses_test, 'ro-', label='Loss(MSE) test')
    plt.plot(range(1,len(losses_train)+1), losses_train, 'ro-', label='Loss(MSE) test')
    plt.title('MSE w.r.t. number of epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss(MSE)')
    plt.legend()
    plt.show()

def correlation_plot(predicted, y):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([min(y), max(y)], [min(y), max(y)], 'r--', lw=4)
    ax.set_xlabel('Original')
    ax.set_ylabel('Predicted')

















