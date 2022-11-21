
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn import train_test_split
from sklearn import preprocessing
from helpers import *
from neural_network import *
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Defining activation function to use
activation = 'sigmoid'

class Customized_dataset(Dataset):

    def __init__(self,X,target):
        super().__init__()
        self.X = torch.tensor(X)
        self.target = torch.tensor(target)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        return self.X[idx,:], self.target[idx]

if __name__ == '__main__':

    gc.collect()

    # Define path for data and useful variable
    name_file = ...
    path = './outputs_test'+ name_file
    batch_size = 128
    lr = 0.1

    # Loading dataset
    X, y, dim_feat = get_single_dataset(path)

    # COnverting data to pytorch dataset object

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 2022)

    # Processing target data depending on activation function

    if activation == 'sigmoid':
        y_train = min_max_scaling(y_train)
        y_test = min_max_scaling(y_test)

    train_dataset = Customized_dataset(X_train,y_train)
    test_dataset = Customized_dataset(X_test,y_test)

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True,
                                num_workers=2, pin_memory=torch.cuda.is_available())

    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=True)
    
    # Cleaning memory

    del train_dataset
    del test_dataset

    gc.collect()

    # Importing model and shift it on GPU (if available)
    model = my_FNN_increasing(batch_size,dim_feat)
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

    for epoch in range(epochs):
        
    ##### TRAINING #####
        
        model.train() # useless since we are not using dropout and batch normalization
        # Defining usegul variables for train_loader
        loss_train_vector = [] #vector of losses for a single epoch

        for batch_idx, (data,target) in enumerate(train_loader):
            if torch.cuda.is_available():
                model.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target).item()
            loss_train_vector.append(loss)
            loss.backward()
            optimizer.step()

        loss_epoch_train.append(np.mean(loss_train_vector))

    ##### TEST #####

        # evaluate model:
        loss_test_vector = [] #vector of losses for a single epoch
        model.eval() 
        lr_history=[]

        for batch_idx, (data,target,y_test) in enumerate(test_loader):
            with torch.no_grad():
                pred = model(data)

            loss = criterion(pred,y_test).item()
            loss_test_vector.append(loss)

        scheduler.step(np.mean(loss_test_vector)) # scheduler step
        lr_history.append(scheduler.get_last_lr()[0])
        
        loss_epoch_test.append(np.mean(loss_test_vector))
    
    visualize_LvsN(loss_epoch_test, loss_epoch_train)











