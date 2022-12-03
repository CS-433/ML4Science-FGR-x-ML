" Some helper function"

import torch 
import torch.nn as nn
import numpy as np
import h5py
from os import listdir
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
        shape : scalar corresponding to the number of features
        mean_halo: float representing the mean of massHalo values
        std_halo: float representing the standard deviation of massHalo values
     """
    f = h5py.File(path)

    # Creating mask

    mass_BH = f['MassBH'][:]
    dot_massBH = f['dotMassBH'][:]
    sfr = f['SFR'][:]

    mask = (mass_BH != 0) & (dot_massBH !=0) & (sfr !=0)

    # Creating structure

    data = np.empty((np.sum(mask), len(features) -1) )

    # Initializing structure

    data[:,0] = f['MassHalo'][mask]  #/ (10**10)
    #data[:,0] = np.log10(data[:,0])
    data[:,1] = f['Nsubs'][mask]
    data[:,2] = f['MassBH'][mask] # / (10**10)
    #data[:,2] = np.log10(data[:,2])
    data[:,3] = f['dotMassBH'][mask] # * 0.978 / (10**10)
    #data[:,3] = np.log10(data[:,3])
    data[:,4] = f['SFR'][mask]

    # Standardizing data

    mean_halo, std_halo = data[:,0].mean(), data[:,0].std()

    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Defining output
    
    target = f['M_HI'][mask]

    return data, target, data.shape[1], mean_halo, std_halo


def get_dataset_LH_fixed(folder_path, features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','VelHalo','z','M_HI']):
    """
    Function to retrieve all the datasets related to a fixed simulation.
    
    Args:
        path: path where to find the file
        features: list of features to extract from the dataset. M_HI must be always passed as last argument
    
    Returns:
        data: array of shape (N,len(features)-1)
        target: array of shape (N,) containing the true output value of each observation
        shape : scalar corresponding to the number of features
        mean_halo: float representing the mean of massHalo values
        std_halo: float representing the standard deviation of massHalo values
     """

    z = [0.77, 0.86, 0.95, 1.05, 1.15, 1.25, 1.36, 1.48, 1.6, 1.73, 1.86, 2, 2.15, 2.3, 2.46, 2.63]

    support_data = {}

    for feature in features:

        # Initializing empty list
        
        support_data[feature] = []

    name_files = [file for file in listdir(folder_path) if not file.startswith('compare')]
    
    for idx,name_file in enumerate(name_files):

        f = h5py.File(folder_path +'/' + name_file)

        dim = f['MassHalo'][:].shape[0]

        for feature in features:

            if feature == 'z':
                support_data[feature].extend( [z[idx]]*dim )

            elif feature == 'VelHalo':
                support_data[feature].extend(np.linalg.norm(f[feature][:], axis = 1))

            else:
                support_data[feature].extend(f[feature][:])
        

    data = np.empty((len(support_data['MassHalo']),len(features)-1))

    # Defining data

    for idx,feature in enumerate(features[:-1]):

        data[:,idx] = np.array(support_data[feature])

    # Defning mask to avoid having too many zero values

    mask = (data[:,2] != 0) & (data[:,3] != 0) & (data[:,4] != 0)

    data = data[mask]

    # Saving mean and std of masHalo for later plots

    mean_halo, std_halo = data[:,0].mean(), data[:,0].std()

    # Standardizing data
    
    data[:,[0,2,3,4,5,6,7,8,9]] = (data[:,[0,2,3,4,5,6,7,8,9]] - np.mean(data[:,[0,2,3,4,5,6,7,8,9]], 
    axis=0)) / (np.std(data[:,[0,2,3,4,5,6,7,8,9]], axis=0))

    target = np.array(support_data['M_HI'], dtype = np.float64)[mask]

    return data,target,data.shape[1], mean_halo, std_halo


##### VISUALIZATION TOOLS #####


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
    axs[1].set(ylim = [-1,1])
    axs[1].legend()

def correlation_plot(predicted, y):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(10**y, 10**predicted, edgecolors=(0, 0, 0))
    ax.plot([min(10**y), max(10**y)], [min(10**y), max(10**y)], 'r--', lw=4)
    ax.set_xlabel('Original')
    ax.set_ylabel('Predicted')
    ax.set_title('Correlation plot: True values vs Predicted values')

def cloud_of_points(predictions,target,massHalo, mean_halo, std_halo):
    """
    Function to plot the MHI against massHalo and compare theroetical results with the predictions of the model."""

    # Converting massHalo values to original scale
    massHalo = (massHalo*std_halo) + mean_halo

    # COnverting output values to original scale

    predictions = 10 ** predictions
    target = 10 ** target

    fig, axs = plt.subplots(1,2, figsize = (10,5))
    axs[0].scatter(massHalo, predictions, alpha = 0.8, marker = '.')
    axs[0].set_title('Scatter plot using predicted data')
    axs[0].set(xscale='log', yscale='log', xlim=(1e9, 1e14), ylim=(1e6, 1e12), xlabel='MassHalo', ylabel='MassHI')

    axs[1].scatter(massHalo,target, alpha = 0.8, marker = '.')
    axs[1].set_title('Scatter plot using original data')
    axs[1].set(xscale='log', yscale='log', xlim=(1e9, 1e14), ylim=(1e6, 1e12), xlabel='MassHalo', ylabel='MassHI' )


















