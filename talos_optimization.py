"""Python file to optimize the architecture"""

import numpy as np
from helpers import *
from sklearn.model_selection import train_test_split
import params
import talos
import pandas as pd

##### GLOBAL ENVIRONMENT #####

# Defining parameters to test during optimization. This dictionary is used by Talos library in order to train different models and return scan_object
p = {'nr_layers':[3,4],
     'hidden_layer_size': [16,32,64],
     'activation': [nn.ReLU(), nn.LeakyReLU()],
     'dropout': [0.05, 0.1],
     'lr': [1e-2,1e-3]}

##### MAIN SCRIPT TO RUN #####

if __name__ == '__main__':
    
    # Loading dataset
    # The function to load data depends on the redshift(s) and simulation(s) one is considering. The optimization process was computed both 
    # considering LH or z fixed, therefore one of the following functions should be used to reproduce the results showed in the paper.
    X, y, dim_feat= get_dataset_LH_fixed('./outputs_test2/LH_0')
    # X, y, dim_feat= get_dataset_z_fixed('./outputs_test2', z = insert_value)

    # Scaling the output. By computing the logarithmic transformation, we want our network to learn the order of magnitude of the Mass_HI
    # and as many digits as possible regarding the mantissa (see Pre-Processing section in the related paper for a better explanation)
    y = np.log10(1 + y)

    # Splitting data into train and test set: 75 % train, 25%
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=42) # we fix the random state for reproducibility purpose

    # Standardizing data
    mean_train, std_train = X_train.mean(axis=0), X_train.std(axis=0)
    X_test, X_train = (X_test - mean_train) / (std_train) , (X_train - mean_train) / (std_train)

    # Computing hyperparameters tuning using talos library
    scan_object = talos.Scan(x=X_train, y=y_train, x_val=X_test, y_val=y_test, params=p, model=optimization_using_talos,
        experiment_name='validation_hyperparameters')










    


















