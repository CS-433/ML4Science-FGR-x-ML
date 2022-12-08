"""Python file to optimize the architecture"""

import numpy as np
from helpers import *
from sklearn.model_selection import train_test_split
import params
import talos
import pandas as pd

##### GLOBAL ENVIRONMENT #####

# Defining parameters to test during optimization
p = {'nr_layers':[2,3,4,5],
     'hidden_layer_size': [16,32,64,128],
     'activation': [nn.ReLU(), nn.LeakyReLU()],
     'dropout': [0.05, 0.1, 0.2],
     'lr': [1e-2,1e-3,1e-4]}

##### MAIN SCRIPT TO RUN #####

if __name__ == '__main__':
    
    # Loading dataset
    # The function to load data depends on the redshift(s) and simulation(s) one is considering
    X, y, dim_feat, mean_halo, std_halo = get_dataset_LH_fixed('./outputs_test2/LH_0')

    # Scaling the output. By computing the logarithmic transformation, we want that our network learns the order of the mass and as many digits as possible regarding its magnitude
    y = np.log10(y)

    # Splitting data into train and test set: 75 % train, 25%
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=2022) # we fix the random state for reproducibility purpose

    # Computing hyperparameters tuning using talos library
    scan_object = talos.Scan(x=X_train, y=y_train, x_val=X_test, y_val=y_test, params=p, model=optimization_using_talos,
        experiment_name='validation_hyperparameters', fraction_limit = 0.008)

    # Be sure that the folder validation_hyperparameters only has one file inside. 
    # The following lines add average train and test loss over the epochs for each model to the result created by talos

    # Retrieving name of created file
    name_file = listdir('./validation_hyperparameters')
    if len(name_file) > 1:
        raise TypeError('More than one file in validation_hyperparameters')
    name_file = name_file[0]

    # Adding average columns
    df = pd.read_csv('./validation_hyperparameters/' + name_file, delimiter = ',')
    df['average_train_loss'] = pd.Series([np.mean(model['loss']) for model in scan_object.round_history])
    df['average_test_loss'] = pd.Series([np.mean(model['val_loss']) for model in scan_object.round_history])
    df = df.sort_values(by = 'average_test_loss', ascending = True)
    df.to_csv('results.csv')









    


















