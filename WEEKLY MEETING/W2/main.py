
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

# Defining activation function to use
activation = 'sigmoid'

class Customized_dataset(Dataset):

    def __init__(self,X,target):
        super.__init__()
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

    X_train,X_test,y_train,y_test = train_test_split(X,y,split = 0.2, random_state = 2022)

    # Processing target data depending on activation function

    if activation == 'sigmoid':
        scaler = preprocessing.MinMaxScaler()
        y_train = scaler.fit_transform(y_train)
        y_test = scaler.fit_transform(y_test)

    train_dataset = Customized_dataset(X_train,y_train)
    test_dataset = Customized_dataset(X_test,y_test)

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True,
                                num_workers=2, pin_memory=torch.cuda.is_available())

    val_loader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=True)
    
    # Cleaning memory

    del train_dataset
    del test_dataset
    gc.collect()

    # Importing model
    model = my_FNN_increasing(batch_size,dim_feat)

    # Defining optimizer
    optimizer = optim.SGD(model.params, lr=lr)

    #Defining loss function
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        criterion.cuda()

    # Defining num_epochs
    epochs = ...

    #Defining useful variables for epochs
    loss_epoch = []
    
    for epoch in range(epochs):
        model.train() # useless since we are not using dropout and batch normalization
        # Defining usegul variables for train_loader
        loss_vector = []
        for batch_idx, (data,target) in enumerate(train_loader):
            if torch.cuda.is_available():
                model.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target).item()
            loss_vector.append(loss)
            loss.backward()
            optimizer.step()
        loss_epoch.append(np.mean(loss_vector))













