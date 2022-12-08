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
        self.__nr_layers = p['nr_layers']
        self.__hidden_layer_size = p['hidden_layer_size']
        self.ending_layer = nn.Linear((self.__nr_layers)*self.__hidden_layer_size, 1, dtype=dtype)
    
    def forward(self,input):

        out = self.dropout(self.activation(self.starting_linear(input)))
        for idx_layer in range(1,self.__nr_layers):
            self.linear = nn.Linear(self.__hidden_layer_size*(idx_layer), self.__hidden_layer_size*(idx_layer+1), dtype = params.dtype).cuda()
            out = self.dropout(self.activation(self.linear(out)))
        out = self.ending_layer(out)
        return out

# Defining function to test neural network with the current set of parameters

def optimization_using_talos(X_train, y_train, X_test, y_test, p):

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

    # Initializing the model
    model = customized_increasing_NN(p,X_train.shape[1],params.dtype)
    if torch.cuda.is_available():
        model = model.cuda()

    # Initializing tools for learning process
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = p['lr'])
    scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 20, min_lr=1e-12, verbose=False)
    
    # Initializing history of the net
    model.init_history()

    # Training the model
    model.train()

    for epoch in range(params.epochs):

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

##### MAIN SCRIPT TO RUN #####

if __name__ == '__main__':
    
    # Loading dataset
    # The function to load data depends on the redshift(s) and simulation(s) one is considering
    X, y, dim_feat, mean_halo, std_halo = get_dataset_LH_fixed('./outputs_test2/LH_0')

    # Scaling the output. By computing the logarithmic transformation, we want that our network learns the order of the mass and as many digits as possible regarding its magnitude
    y = np.log10(y)

    # Splitting data into train and test set: 75 % train, 25%
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=2022) # we fix the random state for reproducibility purpose

    # COmputing hyperparameters tuning using talos library

    scan_object = talos.Scan(x=X_train, y=y_train, x_val=X_test, y_val=y_test, params=p, model=optimization_using_talos,
        experiment_name='Optimization_of_network', fraction_limit = 0.01)

    scan_object.data.to_csv('./results.csv')









    


















