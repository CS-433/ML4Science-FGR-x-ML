" Some helper function"

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import h5py
from os import listdir
import gc
import params
from neural_network import *

# Defining useful class to convert data in a format accepted by Pytorch. 

class Customized_dataset(Dataset):
    """ 
    This class allows to use DataLoader method from Pytorch to create train and test data.
    As described in Pytorch documentation, implementing __len__ and __getitem__ magic methods is required.
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

def get_single_dataset(path, features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','Flux','Density','Temp','VelHalo','M_HI'], masking=True):
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
     """

    # Importing the file 
    f = h5py.File(path)

    # Creating data structure to later store the data
    data = np.empty((f['M_HI'][:].shape[0], len(features) -1) )

    # Collecting data
    for idx,feature in enumerate(features[:-1]):

        if feature == 'VelHalo':
            # Computing the euclidean norm of the velocity
            data[:,idx] = np.linalg.norm(f[feature][:], axis = 1)
        else:
            data[:,idx] = f[feature][:]
    
    # Defining mask to avoid having too many zero values. The mask filters all halos having mass values larger thant 1e10
    if masking:
        mask = (data[:,2] != 0) & (data[:,3] != 0) & (data[:,4] != 0) # the masking is done w.r.t. MassBH, dotMassBH and SFR values
        data = data[mask]

    # Computing log transform of features having skewed distributions
    data[:,[0,7,8]] = np.log(1 + data[:,[0,7,8]])

    # Collecting output values with the appropriate data type
    target = np.array(f['M_HI'][:], dtype = np.float64)
    
    if masking:
        target=target[mask]

    # Closing file
    f.close()

    return data, target, data.shape[1]


def get_dataset_LH_fixed(folder_path, features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','Flux','Density','Temp','VelHalo','z','M_HI'], masking=True):
    """
    Function to retrieve all the datasets related to a precise set of simulations (fixed LH, different redshifts). Since all the observations share the same astrophysical and cosmological
    constants, these values are not considered as features since they would provide useless information.
    
    Args:
        path: string indicating where to find the file
        features: list of features to extract from the dataset. M_HI must be always passed as last argument
        masking : boolean indicating whether to remove zeros values or not. If true, the model is trained only considering halos whose mass is higher than 1e10
    
    Returns:
        data: array of shape (N,len(features)-1)
        target: array of shape (N,) containing the true output value of each observation
        shape : scalar corresponding to the number of features
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
                # Since we have observations with different redshifts, we add this value to the features we consider to train the model
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

    # Defining mask to avoid having too many zero values. The mask filters all halos having mass values larger thant 1e10
    if masking:
        mask = (data[:,2] != 0) & (data[:,3] != 0) & (data[:,4] != 0) # the masking is done w.r.t. MassBH, dotMassBH and SFR values
        data = data[mask]

    # Computing log transformation of features having skewed distributions
    data[:,[0,7,8]] = np.log(1 + data[:,[0,7,8]])

    # Collecting output values with the appropriate data type
    target = np.array(support_data['M_HI'], dtype = np.float64)
    
    if masking:
        target=target[mask]

    return data,target,data.shape[1]


def get_dataset_z_fixed(folder_path, features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','Flux','Density','Temp','VelHalo', 'M_HI'], z = 0.950, masking=True):
    """
    Function to retrieve all the datasets related to a fixed set of simulations (different LH, fixed z). Since z is fixed, this parameter is not considered as a feature 
    since it would provide useless information.

    Args:
        path: string indicating where to find the file
        features: list of features to extract from the dataset. M_HI must be always passed as last argument
        z: scalar corresponding to the analyzed redshift
        masking : boolean indicating whether to remove zeros values or not. If true, the model is trained only considering halos whose mass is higher than 1e10

    Returns:
        data: array of shape (N,len(features)-1)
        target: array of shape (N,) containing the true output value of each observation
        shape : scalar corresponding to the number of features
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

    # Defining mask to avoid having too many zero values. THe mask filters all halos having mass values larger thant 1e10
    if masking:
        mask = (data[:, 2] != 0) & (data[:, 3] != 0) & (data[:, 4] != 0) # the masking is done w.r.t. MassBH, dotMassBH and SFR values
        data = data[mask]

    # Computing log transformation of features having skewed distributions
    data[:,[0,7,8]] = np.log(1 + data[:,[0,7,8]])

    # Collecting output values with appropriate data type
    target = np.array(support_data['M_HI'], dtype=np.float64)[mask] if masking else np.array(support_data['M_HI'], dtype=np.float64)

    return data, target, data.shape[1]


def get_all_dataset(folder_path, features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','Flux','Density','Temp','VelHalo', 'z', 'M_HI'], masking=True):
    """
    Function to retrieve all the data obtained from the simulations. All the possible values for LH and redshift parameters are considered.
    
    Args:
        path: string indicating where to find the file
        features: list of features to extract from the dataset. M_HI must be always passed as last argument
        masking : boolean indicating whether to remove zeros values or not. If true, the model is trained only considering halos whose mass is higher than 1e10
    
    Returns:
        data: array of shape (N,len(features)-1)
        target: array of shape (N,) containing the true output value of each observation
        shape : scalar corresponding to the number of features
     """

    # Checking consistency of the input
    if features[-1] != 'M_HI':
        raise Exception

    # In addition to the input features, we also consider cosmological and astrophysical constants used to obtain the simulated data collected in the files in folder path
    astro_consts = ['Om0', 'sigma8', 'Asn1', 'Aagn1', 'Asn2', 'Aagn2']
    z = [0.77, 0.86, 0.95, 1.05, 1.15, 1.25, 1.36, 1.48, 1.6, 1.73, 1.86, 2, 2.15, 2.3, 2.46, 2.63]
    # We load the above mentioned constants
    params = np.loadtxt('./outputs_test2/params_IllustrisTNG.txt')
    features.extend(astro_consts)

    support_data = {}

    for feature in features:
        # Initializing empty list
        support_data[feature] = []

    # Saving all the file names corresponding to simulations
    name_LH_folders = [LH_folder for LH_folder in listdir(folder_path) if LH_folder.startswith('LH')]

    for idx_LH, name_LH_folder in enumerate(name_LH_folders):
        # After retrieving the name of all the simulations, we create the name of the files we are interesting in by appending the redshift to the name of the simulation
        name_files = [file for file in listdir(folder_path + '/' + name_LH_folder) if file.startswith('MHI')]

        for idx_z, name_file in enumerate(name_files):
            # Importing data from the current file
            f = h5py.File(folder_path + '/' + name_LH_folder + '/' + name_file)
            dim = f['MassHalo'][:].shape[0]

            for idx, feature in enumerate(features):
                if feature in astro_consts:
                    # Since we have observations coming from different simulations, we add for each datapoint the cosmological and astrophysical constants used during the simulation
                    support_data[feature].extend([params[idx_LH][idx - features.index('Om0')]] * dim)

                elif feature == 'VelHalo':
                    # Computing the euclidean norm of the velocity
                    support_data[feature].extend(np.linalg.norm(f[feature][:], axis=1))

                elif feature == 'z':
                    # Since we have observations with different redshifts, we add this value to the feature we consider to train the model
                    support_data[feature].extend([z[idx_z]] * dim)

                else:
                    support_data[feature].extend(f[feature][:])
            # Closing file
            f.close()


    # Defining data
    data = np.empty((len(support_data['MassHalo']), len(features) - 1))

    features.remove('M_HI')

    # Appending all the data from different files in the same numpy array
    for idx, feature in enumerate(features):
        data[:, idx] = np.array(support_data[feature])

    # Defining mask to avoid having too many zero values. THe mask filter all halos having mass values larger thant 1e10
    if masking:
        mask = (data[:, 2] != 0) & (data[:, 3] != 0) & (
                    data[:, 4] != 0)  # the masking is done w.r.t. MassBH, dotMassBH and SFR values
        data = data[mask]

    # Computing log transformation of features with skewed distributions
    data[:,[0,7,8]] = np.log10(1 + data[:,[0,7,8]])

    # Collecting output values with appropriate data type
    target = np.array(support_data['M_HI'], dtype=np.float64)[mask] if masking else np.array(support_data['M_HI'],
                                                                                             dtype=np.float64)

    return data, target, data.shape[1]


def optimization_using_talos(X_train, y_train, X_test, y_test, p):
    """
    Function to perform cross validation on the neural network to find the best hyperparameters.
    This function is used in talos_optimization.py.
    
    Args:
        X_train : numpy array of size (N,D) containing train data
        y_train : numpy array of size (N,) containing train label
        X_test  : numpy array of size (K,D) containing test data
        y_test  : numpy array of size (K,) containing test label
        p : dictionary containing the hyperparameters used to define the network inside the function
        
    Returns:
        model : trained model
        parameters: model parameters
    """

    # To better train the model, instead of using the whole train data at the same time, we divide it in batches using a fixed batch size
    # Converting data into pytorch dataset object
    gc.collect()

    train_dataset = Customized_dataset(X_train,y_train)
    test_dataset = Customized_dataset(X_test,y_test)

    # Divide train and test data into iterable batches
    train_loader = DataLoader(dataset=train_dataset,batch_size=params.batch_size, shuffle=True,
                                num_workers=2, pin_memory=torch.cuda.is_available())

    test_loader = DataLoader(dataset=test_dataset,batch_size=params.batch_size, shuffle=True,
                                pin_memory=torch.cuda.is_available())
    
    del train_dataset
    del test_dataset

    gc.collect()

    # Initializing the model. The following model depends on the values contained in the dictionary passed as an input
    model = customized_increasing_NN(p,X_train.shape[1],params.dtype)
    if torch.cuda.is_available():
        model = model.cuda()

    # Initializing tools to use during learning process
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = p['lr'])
    # Defining scheduler, in order to reduce the learning rate when needed and speed up the training procedure
    scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 20, min_lr=1e-12, verbose=False)
    
    # Initializing history of the net
    model.init_history()

    # Training the model
    model.train()

    for _ in range(params.epochs):

        # Defining support data structure
        loss_train, loss_test = [], []

        for (data,target) in train_loader:

            # Moving test data to GPU if possible
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # Training and updating weights
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            # Storing current loss value in a provisional list
            loss_train.append(loss.item())
        
        # Storing the current loss value using talos history
        model.append_loss(np.mean(loss_train))

        # Using test data
        for (data,target) in test_loader:

            # Moving test data to GPU if possible
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                # Storing current loss value in a provisional list
                loss_test.append(loss.item())
                
        # Storing the current loss value using talos history
        model.append_val_loss(np.mean(loss_test))

    return model, model.parameters()


















