import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from helpers import *
from neural_network import *
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os import makedirs
from shutil import rmtree
import params
from sklearn.metrics import r2_score
import pickle

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

##### MAIN SCRIPT TO RUN #####
if __name__ == '__main__':

    gc.collect()
    
    # Loading dataset
    
    #X, y, dim_feat = get_single_dataset(params.path_file)
    X, y, dim_feat = get_dataset_LH_fixed('./outputs_test2/LH_0/', features = ['MassHalo','Nsubs','MassBH','dotMassBH','SFR','Flux','Density','Temp','VelHalo','z','M_HI'] )

    # Scaling the output
    y = np.log10(y)

    # Splitting data into train and test set
    # 75 % train, 20% test, 5% validation
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=2022)

    X_test,X_val,y_test,y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=2022)

    # Converting data into pytorch dataset object
    train_dataset = Customized_dataset(X_train,y_train)
    test_dataset = Customized_dataset(X_test,y_test)

    # Since we do not want to iterate over the validation set, we only convert it to a tensor
    X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
    if torch.cuda.is_available():
        X_val = X_val.cuda()
        y_val = y_val.cuda()

    # Divide train and test data into iterable batches
    train_loader = DataLoader(dataset=train_dataset,batch_size=params.batch_size, shuffle=True,
                                num_workers=2, pin_memory=torch.cuda.is_available())

    test_loader = DataLoader(dataset=test_dataset,batch_size=params.batch_size, shuffle=True,
                                pin_memory=torch.cuda.is_available())
    
    # Cleaning memory
    del train_dataset
    del test_dataset

    gc.collect()

#-----------------------------------------------------------------------------------------------
    
    # Defining num_epochs
    epochs = params.epochs

    #Defining loss function and shift it on GPU (if available)
    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        criterion.cuda()

    # First run case
    if (params.first_run == True):
        
        # creation of the folder "checkpoints"
        makedirs('./checkpoints/', exist_ok=True)
        rmtree('./checkpoints/') # to remove all previous checkpoint files
        makedirs('./checkpoints/') 
    
        #Defining useful variables for epochs
        loss_epoch_train = [] # will contain all the train losses of the different epochs
        loss_epoch_test = [] # will contain all the test losses of the different epochs
        R2_epoch_test = []
        R2_epoch_train = []
        
        # Initializing number epoch and loss
        final_epoch=epochs
        prev_loss=torch.inf
        current_epoch=0

        # Importing model and move it on GPU (if available)
        model = my_FNN_increasing(dim_feat,params.dtype)
        if(torch.cuda.is_available()): # for the case of laptop with local GPU
            model = model.cuda()

        # Defining optimizer
        optimizer = optim.SGD(model.parameters(), lr=params.lr)

        # Defining a scheduler to adjust the learning rate
        scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 20, min_lr=1e-12, verbose=True)

    else:

        # Resuming the training from last_model
        PATH = './checkpoints/last_model.pt'

        model = my_FNN_increasing(dim_feat,params.dtype)

        # Defining optimizer
        optimizer = optim.SGD(model.parameters(), lr=params.lr)

        # Defining a scheduler to adjust the learning rate
        scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 20, min_lr=1e-12, verbose=True)

        checkpoint = torch.load(PATH)

        # Importing final state of the model from previous run
        model.load_state_dict(checkpoint['model_state'])

        if(torch.cuda.is_available()): # for the case of laptop with local GPU
            model = model.cuda()

        # Importing final state of the optimizer from previous run
        optimizer.load_state_dict(checkpoint['optimizer_state'])

        # Importing final state of the scheduler from previous run. By doing so, we keep trace of previous learning rates.
        scheduler.load_state_dict(checkpoint['scheduler_state'])

        # Defining current epoch
        current_epoch = checkpoint['epoch'] + 1   # since we save the last epoch done, we have to start from the correct one

        # Resuming previous loss
        prev_loss = checkpoint['prev_loss']

        final_epoch = current_epoch + epochs # updating the number of the final epoch
    
        # Resuming loss vectors defined before
        loss_epoch_train = pickle.load(open("./checkpoints/loss_train.txt", "rb"))  # to load the vector of train losses
        loss_epoch_train = loss_epoch_train['train_loss']
        loss_epoch_test = pickle.load(open("./checkpoints/loss_test.txt", "rb"))    # to load the vector of test losses
        loss_epoch_test = loss_epoch_test['test_loss']

        # Resuming R2 vectors defined before
        R2_train = pickle.load(open("./checkpoints/R2_train.txt","rb")) # to load the vector of train R2s
        R2_test = pickle.load(open("./checkpoints/R2_test.txt", "rb")) # to load the vector of test R2s
        R2_epoch_train = R2_train['R2_train']
        R2_epoch_test = R2_test['R2_test']
#-----------------------------------------------------------------------------------------------------------------------------------
    
    for epoch in range(current_epoch, final_epoch):
        
        ##### TRAINING #####

        model.train() # useless since we are not using dropout and batch normalization

        loss_train_vector = [] #vector of losses for a single epoch
        R2_train = []

        for batch_idx, (data,target) in enumerate(train_loader):
            # Moving data to the GPU if possible
            if(torch.cuda.is_available()): # for the case of laptop with local GPU
                data,target = data.cuda(), target.cuda()
            # Setting the gradient attribute of each weight to zero
            optimizer.zero_grad()
            # Computing the forward pass
            output = model(data)
            # Computing the loss
            loss = criterion(torch.flatten(output),target)
            #print('Loss: ', loss)
            #print('Output: ', output)
            #print('Target: ', target)
            # Computing the gradient w.r.t. model parameters
            loss.backward()
            # Adjusting the weights using SGD
            optimizer.step()
            # Saving the loss in the corresponding vector
            loss_train_vector.append(loss.item())
            # Computing the R2 score
            R2 = r2_score(target.cpu().detach().numpy(), output.cpu().detach().numpy().squeeze()) # computing R2 score
            # Appending result
            R2_train.append(R2) # storing R2 score

        loss_train = np.mean(loss_train_vector)
        # Comparing the loss of the epoch with the previous ones to check whether to change the learning rate or not
        scheduler.step(loss_train) 
        # Saving the train loss of the current epoch for later plot
        loss_epoch_train.append(loss_train)

        R2_train = np.mean(R2_train)
        R2_epoch_train.append(R2_train)
        
        # Saving the loss in an apposite file
        pickle.dump({"train_loss": loss_epoch_train}, open("./checkpoints/loss_train.txt", "wb")) # it overwrites the previous file
        np.savetxt('./checkpoints/R2_train.txt', np.array(R2_epoch_train))

        ##### TEST #####

        model.eval() 

        loss_test_vector = [] #vector of losses for a single epoch
        R2_test = []

        for batch_idx, (data,target) in enumerate(test_loader):
            
            # Moving data to the GPU if possible
            if(torch.cuda.is_available()): # for the case of laptop with local GPU
                data,target = data.cuda(), target.cuda()
            # Using torch.no_grad() to not save the operation in the computation graph
            with torch.no_grad():
                pred = model(data)
            # Computing test loss. Since pred.require_grad = False, the following operation is not added to the computation graph
            loss = criterion(torch.flatten(pred),target).item()
            loss_test_vector.append(loss)
            R2 = r2_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy()) # computing R2 score
            R2_test.append(R2) # storing R2 score


        # Saving the test loss of the current epoch for later plot
        loss_test = np.mean(loss_test_vector)
        loss_epoch_test.append(loss_test)

        R2_test = np.mean(R2_test)
        R2_epoch_test.append(R2_test)

        # Visualizing loss values against the number of epoch
        if (epoch+1)%5 == 0 and epoch != 0:
            visualization(loss_epoch_test, loss_epoch_train, R2_epoch_train, R2_epoch_test)
            plt.savefig('./checkpoints/visualization_plot.png', bbox_inches='tight') # saving plot

        # Saving the loss and the R2 score in an apposite file
        pickle.dump({"test_loss": loss_epoch_test}, open("./checkpoints/loss_test.txt", "wb")) # it overwrites the previous file
        np.savetxt('./checkpoints/R2_test.txt', np.array(R2_epoch_test))
        
        # If we get a better model, save it. Therefore this file will contain the best model so far
        if (loss_test < prev_loss): # if our current model is better, update the best model saving the net state, loss value and R2 score
            prev_loss = loss_test
            PATH = './checkpoints/best_model.pt'
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'best_loss': prev_loss}, PATH)

        print('Epoch %d: train_loss=%.4f, test_loss=%.4f' %(epoch+1, loss_train, loss_test))

        # Saving the last model used at evey epoch
        PATH = './checkpoints/last_model.pt'
        torch.save({'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'prev_loss': prev_loss}, PATH)
    
    ##### VALIDATION #####

    PATH = './checkpoints/best_model.pt'

    best_model = my_FNN_increasing(dim_feat,params.dtype)

    checkpoint = torch.load(PATH)

    # Importing final state of the model from previous run
    best_model.load_state_dict(checkpoint['model_state'])

    if(torch.cuda.is_available()): # for the case of laptop with local GPU
        best_model = best_model.cuda()

    # After importing the model, we just need to compute the prediction on validation data

    with torch.no_grad():

        output_validation = best_model(X_val)
        
    plt.clf() # to clear the current figure
    correlation_plot(output_validation.cpu().detach().numpy(), y_val.cpu().detach().numpy())
    plt.savefig('./checkpoints/correlation_plot.png', bbox_inches='tight') # saving correlation plot







