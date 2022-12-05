"""Python file to optimize the architecture"""

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from helpers import *
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os import makedirs
from shutil import rmtree
import params
import talos 
from talos.utils import lr_normalizer



##### GLOBAL ENVIRONMENT #####

# Defining useful class to convert data in a format accepted by Pytorch

class Customized_dataset(Dataset):

    def __init__(self,X,target):
        super().__init__()
        self.X = torch.tensor(X)
        self.target = torch.tensor(target)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        return self.X[idx,:], self.target[idx]

# Defining parameters to test during optimization

p = {'nr_layers':[2,3,4,5],
     'hidden_layer_size': [16,32,64,128],
     'activation': [nn.ReLU(), nn.LeakyReLU()],
     'dropout': [0.05, 0.1, 0.2],
     'lr': [1e-2,1e-3,1e-4]}

# Defining neural network architecture to use Talos optimization

class customized_increasing_NN(nn.Module, talos.utils.TorchHistory):

    def __init__ (self,p, num_features,dtype):
        super().__init__()
        # We use linear layer, we use params to define other layers
        self.dropout = nn.Dropout(p['dropout'])
        self.activation = p['activation']
        self.starting_linear = nn.Linear(num_features, p['hidden_layer_size'], dtype=dtype)
    
    def forward(self,input):

        out = self.dropout(self.activation(self.starting_linear(input)))

        for idx_layer in range(1,p['nr_layers']):
            self.linear = nn.Linear(p['hidden:layer_size']*(idx_layer), p['hidden:layer_size']*(idx_layer+1))
            out = self.dropout(self.activation(self.linear(out)))
        return out

# Defining function to test neural network with the current set of parameters

def optimization_using_talos(X_train, y_train, X_test, y_test, p):

    # Initializing the model
    model = customized_increasing_NN(p,X_train.shape[1],params.dtype)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = p['lr'])
    scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 20, min_lr=1e-12, verbose=True)
    # Initializing history of the net
    model.init_history()

    # INitializing list to store train loss

    # Training the model
    for epoch in range(params.epochs):

        # Training and updating weights
        optimizer.zero_grad()
        model.train()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        # Storing the current loss value using talos history
        model.append_loss(loss.item())

        # Using validation data
        with torch.no_grad():
            output = model(X_test)
            loss = criterion(output,y_test)
            model.append_val_loss(loss.item())

    return model, model.parameters()

##### MAIN SCRIPT TO RUN #####

if __name__ == '__main__':

    gc.collect()
    
    # Loading dataset
    
    #X, y, dim_feat = get_single_dataset(params.path_file)
    X, y, dim_feat, mean_halo, std_halo = get_dataset_LH_fixed('./outputs_test2/LH_0', 
                                        features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','Flux','Density','Temp','VelHalo','z','M_HI'],
                                        masking = True)

    # Scaling the output
    y = np.log10(y)

    # Splitting data into train and test set
    # 75 % train, 20% test, 5% validation
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=2022)

    X_test,X_val,y_test,y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=2022)

    # Converting data into pytorch dataset object
    train_dataset = Customized_dataset(X_train,y_train)

    test_dataset = Customized_dataset(X_test,y_test)


    scan_object = talos.Scan(x=X_train, y=y_train, x_val=X_test, y_val=y_test, params=p, model=optimization_using_talos,
        experiment_name='Optimization_of_network', round_limit=100)








    


















