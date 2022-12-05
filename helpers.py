" Some helper function"

import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import h5py
from os import listdir

# Defining useful class to convert data in a format accepted by Pytorch. 

class Customized_dataset(Dataset):
    """ 
    This class allows to use DataLoader method from Pytorch to create train and test data
    """

    def __init__(self,X,target):
        super().__init__()
        # Converting numpy object to tensor
        self.X = torch.tensor(X)
        self.target = torch.tensor(target)
    
    def __len__(self):
        # The magic method __len__ must be implemented in order to use Pytorch DataLoader
        return self.X.shape[0]

    def __getitem__(self, idx):
        # The magic method __getitem__ must be implemented in order to use Pytorch DataLoader
        return self.X[idx,:], self.target[idx]


##### HELPER FUNCTIONS #####

def get_single_dataset(path, features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','Flux','Density','Temp','VelHalo','z','M_HI'], masking=True):
    """
    Function to retrieve a single dataset. The corresponding simulation (LH) and redshift (z) are given as input in the path string.
    
    Args:
        path: string indicating where to find the file
        features: list of features to extract from the dataset. M_HI must be always passed as last argument
        masking : boolean indicating whether to remove zeros values or not. If true, the model is trained only considering halos whose mass is higher than 1e10
    
    Returns:
        data: array of shape (N,len(features)-1)
        target: array of shape (N,) containing the true output value of each observation
        shape : scalar corresponding to the number of features
        mean_halo: float representing the mean of massHalo values
        std_halo: float representing the standard deviation of massHalo values
     """

    # Importing the file 
    f = h5py.File(path)

    # Creating data structure to later store the data
    data = np.empty((np.sum(mask), len(features) -1) )

    # Collecting data
    for idx,feature in features[:-1]:

        if feature == 'VelHalo':
            # Computing the euclidean norm of the velocity
            data[:,idx] = np.linalg.norm(f[feature][:], axis = 1)
        else:
            data[:,idx] = f[feature][:]
    
    # Defining mask to avoid having too many zero values. THe mask filter all halos having mass values larger thant 1e10
    if masking:
        mask = (data[:,2] != 0) & (data[:,3] != 0) & (data[:,4] != 0) # the masking is done w.r.t. MassBH, dotMassBH and SFR values
        data = data[mask]

    # Saving mean and std of massHalo for later plots(for further details, see plots.py)
    mean_halo, std_halo = data[:,0].mean(), data[:,0].std()

    # Standardizing data
    features_to_standardize = [0,2,3,4,5,6,7,8,9]
    data[:, features_to_standardize] = (data[:, features_to_standardize] - np.mean(data[:, features_to_standardize], axis=0)) / np.std(data[:, features_to_standardize], axis=0)

    # Collecting output values
    target = f['M_HI'][mask] if masking else f['M_HI'][:]
    target.dtype = np.float64

    # Closing file
    f.close()

    return data, target, data.shape[1], mean_halo, std_halo


def get_dataset_LH_fixed(folder_path, features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','Flux','Density','Temp','VelHalo','z','M_HI'], masking=True):
    """
    Function to retrieve all the datasets related to a fixed simulation (fixed LH, different redshifts). Since all the observations share the same astrophysical and cosmological
    constants, these values are not considered as features since they would provide useless information.
    
    Args:
        path: string indicating where to find the file
        features: list of features to extract from the dataset. M_HI must be always passed as last argument
        masking : boolean indicating whether to remove zeros values or not. If true, the model is trained only considering halos whose mass is higher than 1e10
    
    Returns:
        data: array of shape (N,len(features)-1)
        target: array of shape (N,) containing the true output value of each observation
        shape : scalar corresponding to the number of features
        mean_halo: float representing the mean of massHalo values
        std_halo: float representing the standard deviation of massHalo values
     """

    # Defining all the redshifts in our simulation
    z = [0.77, 0.86, 0.95, 1.05, 1.15, 1.25, 1.36, 1.48, 1.6, 1.73, 1.86, 2, 2.15, 2.3, 2.46, 2.63]

    support_data = {}

    for feature in features:

        # Initializing empty list
        support_data[feature] = []

    # Saving all the file names corresponding to LH given in folder_path
    name_files = [file for file in listdir(folder_path) if not file.startswith('compare')]
    
    for idx,name_file in enumerate(name_files):

        # Importing data from the current file
        f = h5py.File(folder_path +'/' + name_file)

        dim = f['MassHalo'][:].shape[0]

        # Saving each feature of the current file in support_data
        for feature in features:

            if feature == 'z':
                # Since we have observations with different redshifts, we add this value to the feature we consider to train the model
                support_data[feature].extend( [z[idx]]*dim )

            elif feature == 'VelHalo':
                # Computing the euclidean norm of the velocity
                support_data[feature].extend(np.linalg.norm(f[feature][:], axis = 1))

            else:
                support_data[feature].extend(f[feature][:])
        f.close()   

    # Defining data
    data = np.empty((len(support_data['MassHalo']),len(features)-1))

    # Appending all the data from different files in the same numpy array
    for idx,feature in enumerate(features[:-1]):

        data[:,idx] = np.array(support_data[feature])

    # Defining mask to avoid having too many zero values. The mask filter all halos having mass values larger thant 1e10
    if masking:
        mask = (data[:,2] != 0) & (data[:,3] != 0) & (data[:,4] != 0) # the masking is done w.r.t. MassBH, dotMassBH and SFR values
        data = data[mask]

    # Saving mean and std of massHalo for later plots(for further details, see plots.py) 
    mean_halo, std_halo = data[:,0].mean(), data[:,0].std()

    # Standardizing data
    features_to_standardize = [0,2,3,4,5,6,7,8,9]
    data[:,features_to_standardize] = (data[:,features_to_standardize] - np.mean(data[:,features_to_standardize], axis=0)) / (np.std(data[:,features_to_standardize], axis=0))

    # Collecting output values
    target = np.array(support_data['M_HI'], dtype = np.float64)
    
    if masking:
        target=target[mask]

    return data,target,data.shape[1], mean_halo, std_halo


def get_dataset_z_fixed(folder_path, features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','Flux','Density','Temp','VelHalo', 'M_HI'], z = 0.950, masking=True):
    """
    Function to retrieve all the datasets related to a fixed redshift (different LH, fixed z). SInce z is fixed, its values is not considered as a feature 
    since it would provide useless information.

    Args:
        path: string indicating where to find the file
        features: list of features to extract from the dataset. M_HI must be always passed as last argument
        z: scalar corresponding to the analyzed redshift

    Returns:
        data: array of shape (N,len(features)-1)
        target: array of shape (N,) containing the true output value of each observation
        shape : scalar corresponding to the number of features
        mean_halo: float representing the mean of massHalo values
        std_halo: float representing the standard deviation of massHalo values
     """

    # In addition to the input features, we also consider cosmological and astrophysical constants used to obtain the simulated data collected in the files in folder path
    astro_consts = ['Om0', 'sigma8', 'Asn1', 'Aagn1', 'Asn2', 'Aagn2']
    # We load the above mentioned constants
    params = np.loadtxt('./outputs_test2/params_IllustrisTNG.txt')
    features.extend(astro_consts)

    support_data = {}

    for feature in features:
        # Initializing empty list
        support_data[feature] = []

    # Saving all the file names corresponding to simulations
    name_LH_folders = [LH_folder for LH_folder in listdir(folder_path) if LH_folder.startswith('LH')]


    for idx_LH,name_LH_folder in enumerate(name_LH_folders):
        # After retrieving the name of all the simulations, we create the name of the files we are interesting in by appending the redshift to the name of the simulation
        sim_file = 'MHI_LH'+ str(idx_LH) + '_z=' + '{:.3f}'.format(z) + '.hdf5'
    
        # Importing data from the current file
        f = h5py.File(folder_path + '/' + name_LH_folder + '/' + sim_file)

        dim = f['MassHalo'][:].shape[0]

        for idx, feature in enumerate(features):
            if feature in astro_consts:
                # Since we have observations coming from different simulations, we add for each datapoint the cosmological and astrophysical constants used during the simulation
                support_data[feature].extend([params[idx_LH][idx - features.index('Om0')]]*dim)

            elif feature == 'VelHalo':
                # Computing the euclidean norm of the velocity
                support_data[feature].extend(np.linalg.norm(f[feature][:], axis=1))

            else:
                support_data[feature].extend(f[feature][:])
        
        # Closing file
        f.close()

    # Defining data
    data = np.empty((len(support_data['MassHalo']),len(features)-1))

    features.remove('M_HI')

    # Appending all the data from different files in the same numpy array
    for idx,feature in enumerate(features):
        data[:,idx] = np.array(support_data[feature])

    # Defining mask to avoid having too many zero values. THe mask filter all halos having mass values larger thant 1e10
    if masking:
        mask = (data[:, 2] != 0) & (data[:, 3] != 0) & (data[:, 4] != 0) # the masking is done w.r.t. MassBH, dotMassBH and SFR values
        data = data[mask]

    # Saving mean and std of massHalo for later plots(for further details, see plots.py)
    mean_halo, std_halo = data[:, 0].mean(), data[:, 0].std()

    # Standardizing data
    features_to_standardize = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    data[:, features_to_standardize] = (data[:, features_to_standardize] - np.mean(data[:, features_to_standardize], axis=0)) / (np.std(data[:, features_to_standardize], axis=0))

    # Collecting output values
    target = np.array(support_data['M_HI'], dtype=np.float64)[mask] if masking else np.array(support_data['M_HI'], dtype=np.float64)

    return data, target, data.shape[1], mean_halo, std_halo


















