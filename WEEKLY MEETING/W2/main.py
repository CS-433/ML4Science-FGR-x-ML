
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from helpers import *
from neural_network import *
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau

##### GLOBAL ENVIRONMENT #####

# Define path for data and useful variables
name_file = 'LH_0/MHI_LH0_z=0.770.hdf5'
path = './outputs_test/'+ name_file
batch_size = 128
lr = 0.1
activation = 'sigmoid'

# Defining useful class to convert data in a format accepted by Pytorch
class Customized_dataset(Dataset):

    def __init__(self,X,target):
        super().__init__()
        self.X = torch.tensor(X)
        self.target = torch.tensor(target)
        self.dtype = self.X.dtype
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        return self.X[idx,:], self.target[idx]

##### MAIN SCRIPT TO RUN #####
if __name__ == '__main__':

    gc.collect()
    
    # Loading dataset
    X, y, dim_feat = get_single_dataset(path)

    # Line to try
    X = X[:1000]
    y = y[:1000]

    # Splitting data into train and test set
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=2022)

    # Processing target data depending on activation function
    if activation == 'sigmoid':
        y_train = min_max_scaling(y_train)
        y_test = min_max_scaling(y_test)
    
    # Converting data into pytorch dataset object
    train_dataset = Customized_dataset(X_train,y_train)
    test_dataset = Customized_dataset(X_test,y_test)

    # Divide train and test data into iterable batches
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True,
                                num_workers=2, pin_memory=torch.cuda.is_available())

    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=True)

    dtype = train_dataset.dtype
    
    # Cleaning memory
    del train_dataset
    del test_dataset

    gc.collect()

    # Importing model and move it on GPU (if available)
    model = my_FNN_increasing(dim_feat,dtype)
    if(torch.cuda.is_available()): # for the case of laptop with local GPU
        model = model.cuda()

    # Defining optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defining a scheduler to adjust the learning rate
    scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 20, min_lr=1e-12, verbose=True)

    #Defining loss function and shift it on GPU (if available)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        criterion.cuda()
        
    # Defining num_epochs
    epochs = 500

    #Defining useful variables for epochs
    loss_epoch_train = [] # will contain all the train losses of the different epochs
    loss_epoch_test = [] # will contain all the test losses of the different epochs
    lr_history = [] #will contain lr after each epoch

    for epoch in range(epochs):
        
        ##### TRAINING #####

        model.train() # useless since we are not using dropout and batch normalization

        loss_train_vector = [] #vector of losses for a single epoch

        for batch_idx, (data,target) in enumerate(train_loader):
            # Moving data to the GPU if possible
            if(torch.cuda.is_available()): # for the case of laptop with local GPU
                data,target = data.cuda(), target.cuda()
            # Setting the gradient attribute of each weight to zero
            optimizer.zero_grad()
            # Computing the forward pass
            output = model(data)
            # Computing the loss
            loss = criterion(output,target)
            # Computing the gradient w.r.t. model parameters
            loss.backward()
            # Adjusting the weights using SGD
            optimizer.step()
            # Saving the loss in the corresponding vector
            loss_train_vector.append(loss.item())

        # Comparing the loss of the epoch with the previous ones to check whether to change the learning rate or not
        scheduler.step(np.mean(loss_train_vector)) 
        # Saving the learning rate in the apposite vector
        lr_history.append(scheduler.get_last_lr()[0])
        # Saving the train loss of the current epoch for later plot
        loss_epoch_train.append(torch.mean(loss_train_vector))

        ##### TEST #####

        model.eval() 

        loss_test_vector = [] #vector of losses for a single epoch

        for batch_idx, (data,target,y_test) in enumerate(test_loader):
            # Moving data to the GPU if possible
            if(torch.cuda.is_available()): # for the case of laptop with local GPU
                data,target = data.cuda(), target.cuda()
            # Using torch.no_grad() to not save the operation in the computation graph
            with torch.no_grad():
                pred = model(data)
            # Computing test loss. Since pred.require_grad = False, the following operation is not added to the computation graph
            loss = criterion(pred,y_test).item()
            loss_test_vector.append(loss)
        
        # Saving the test loss of the current epoch for later plot
        loss_epoch_test.append(np.mean(loss_test_vector))
    
    # Visualizing loss values against the number of epoch
    visualize_LvsN(loss_epoch_test, loss_epoch_train)











