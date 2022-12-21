import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from helpers import *
from plots import *
from neural_network import *
import gc
from os import makedirs
from shutil import rmtree
import params
import pickle


##### MAIN SCRIPT TO RUN #####

if __name__ == '__main__':

#------- INITIALIZATION AND IMPORTING DATA -------#

    gc.collect()
    
    # Loading dataset
    # The function to load data depends on the redshift(s) and simulation(s) one is considering
    # Please refer to helpers.py to know how to retrieve data depending on the choice of these parameters
    X, y, dim_feat = get_all_dataset('./outputs_test2', masking=False)

    # Scaling the output. By computing the logarithmic transformation, we want our network to learn the order of magnitude of M_HI and 
    # as many digits as possible regarding the mantissa. Notice that we add 1 to the target in order to avoid problems with small values
    y = np.log10(1 + y)

    # Splitting data into train and test set: 75 % train, 20% test, 5% validation
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=42) # we fix the random state for reproducibility purpose
    X_test,X_val,y_test,y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

    # Standardizing data
    mean_train, std_train = X_train.mean(axis=0), X_train.std(axis=0)
    X_test, X_train = (X_test - mean_train) / (std_train) , (X_train - mean_train) / (std_train)
    X_val = (X_val - mean_train) / std_train

    # Converting data into pytorch dataset object
    train_dataset = Customized_dataset(X_train,y_train)
    test_dataset = Customized_dataset(X_test,y_test)

    # Since we do not want to iterate over the validation set, we only convert it to a tensor
    X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
    if torch.cuda.is_available(): # if a GPU is available, we move the validation data to this device to later speed up the validation process
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

#------- IMPORTING AN ALREADY TRAINED MODEL (IF FIRST_RUN = FALSE) OR INITIALIZING TOOLS TO TRAIN THE NETWORK (IF FIRST_RUN = TRUE) -------#
    
    # Defining num_epochs
    epochs = params.epochs

    # Defining loss function and move it to GPU (if available)
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
        loss_epoch_train = [] # it will contain the train losses of the different epochs
        loss_epoch_test = [] # it will contain the test losses of the different epochs
        R2_epoch_test = [] # it will contain R2 score on test data of the different epochs
        R2_epoch_train = [] # it will contain R2 score on train data of the different epochs
        
        # Initializing number of epochs and loss
        final_epoch=epochs
        prev_loss=torch.inf
        current_epoch=0

        # Importing model and move it to GPU (if available)
        # The choice of the model should reflect the way we load the data (masking or non masking model depending on the boolean value given as
        # input when retrieving data at the beginning of the notebook)

        if params.masking == True:
            model = my_FNN_increasing_masking(dim_feat,params.dtype)
        else:
             model = my_FNN_increasing_NOmasking(dim_feat,params.dtype)

        if(torch.cuda.is_available()): # for the case of device with local GPU
            model = model.cuda()

        # Defining optimizer (Stochastic Gradient Descent)
        optimizer = optim.SGD(model.parameters(), lr=params.lr)

        # Defining a scheduler to adjust the learning rate in case of slow learning process
        scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 20, min_lr=1e-12, verbose=True)

    # If a model has already been trained (and therefore provisional results are contained in checkpoints folder).
    # This is particularly useful when training the model for a large number of epochs (you can divide the problem in subruns) or when using 
    # an external cluster

    else:

        # Resuming the training from last_model
        PATH = './checkpoints/last_model.pt'

        if params.masking == True:
            model = my_FNN_increasing_masking(dim_feat,params.dtype)
        else:
            model = my_FNN_increasing_NOmasking(dim_feat,params.dtype)

        # Defining optimizer
        optimizer = optim.SGD(model.parameters(), lr=params.lr)

        # Defining a scheduler to adjust the learning rate
        scheduler = ReduceLROnPlateau(optimizer = optimizer, mode = 'min', factor = 0.1, patience = 20, min_lr=1e-12, verbose=True)

        # Load previous results in order to restore them
        checkpoint = torch.load(PATH)

        # Importing final state of the model from previous run
        model.load_state_dict(checkpoint['model_state'])

        # Importing model and move it to GPU (if available)
        if(torch.cuda.is_available()): # for the case of device with local GPU
            model = model.cuda()

        # Importing final state of the optimizer from previous run
        optimizer.load_state_dict(checkpoint['optimizer_state'])

        # Importing final state of the scheduler from previous run. By doing so, we keep trace of previous learning rates.
        scheduler.load_state_dict(checkpoint['scheduler_state'])

        # Defining current epoch
        current_epoch = checkpoint['epoch'] + 1   # since we save the last epoch that has been done, we have to start from the correct one

        # Retrieving previous loss
        prev_loss = checkpoint['prev_loss']

        # Defining final epoch
        final_epoch = current_epoch + epochs # updating the number of the final epoch
    
        # Retrieving loss vectors defined before
        loss_epoch_train = pickle.load(open("./checkpoints/loss_train.txt", "rb"))  # to load the vector of train losses
        loss_epoch_train = loss_epoch_train['train_loss']
        loss_epoch_test = pickle.load(open("./checkpoints/loss_test.txt", "rb"))    # to load the vector of test losses
        loss_epoch_test = loss_epoch_test['test_loss']

        # Retrieving R2 vectors defined before
        R2_train = pickle.load(open("./checkpoints/R2_train.txt","rb")) # to load the vector of train R2s
        R2_test = pickle.load(open("./checkpoints/R2_test.txt", "rb"))  # to load the vector of test R2s
        R2_epoch_train = R2_train['R2_train']
        R2_epoch_test = R2_test['R2_test']

#------- TRAINING AND TESTING PROCESS -------#
    
    for epoch in range(current_epoch, final_epoch):
        
        ##### TRAINING #####

        model.train() # useful when using Dropout

        loss_train_vector = [] #vector of losses on train data for a single epoch
        R2_train = []          #vector of R2 scores on train data for a single epoch

        for batch_idx, (data,target) in enumerate(train_loader):

            # Moving data to the GPU if possible
            if(torch.cuda.is_available()): # for the case of device with local GPU
                data,target = data.cuda(), target.cuda()

            # Setting the gradient attribute of each weight to zero
            optimizer.zero_grad()
            # Computing the forward pass
            output = model(data)
            # Computing the loss
            loss = criterion(torch.flatten(output),target)
            # Computing the gradient w.r.t. model parameters
            loss.backward()
            # Adjusting the weights using SGD
            optimizer.step()
            # Saving the loss in the corresponding vector
            loss_train_vector.append(loss.item())
            # Computing the R2 score
            R2 = r2_score(target.cpu().detach().numpy(), output.cpu().detach().numpy().squeeze()) 
            # Appending result
            R2_train.append(R2) # storing R2 score

        # Storing mean of loss values
        loss_train = np.mean(loss_train_vector)

        # Comparing the loss of the epoch with the previous ones to check whether to change the learning rate or not using the scheduler
        scheduler.step(loss_train) 

        # Saving the train loss of the current epoch for later plot
        loss_epoch_train.append(loss_train)

        # Saving the R2 scores of the current epoch for later plot
        R2_train = np.mean(R2_train)
        R2_epoch_train.append(R2_train)
        
        # Saving the loss in a file
        pickle.dump({"train_loss": loss_epoch_train}, open("./checkpoints/loss_train.txt", "wb")) # it overwrites the previous file
        np.savetxt('./checkpoints/R2_train.txt', np.array(R2_epoch_train))

        ##### TEST #####

        model.eval() # useful when using Droput

        loss_test_vector = [] # vector of losses on test data for a single epoch
        R2_test = []          # vector of R2 scores on test data for a single epoch

        for batch_idx, (data,target) in enumerate(test_loader):
            
            # Moving data to the GPU if possible
            if(torch.cuda.is_available()): # for the case of laptop with local GPU
                data,target = data.cuda(), target.cuda()
            # Using torch.no_grad() to not save the operation in the computation graph
            with torch.no_grad():
                pred = model(data)
            # Computing test loss. Since pred.require_grad = False, the following operation is not added to the computation graph
            loss = criterion(torch.flatten(pred),target).item()
            # Saving the loss in the corresponding vector
            loss_test_vector.append(loss)
            # Computing the R2 score
            R2 = r2_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy()) 
            # Appending result
            R2_test.append(R2) # storing R2 score

        # Storing mean of loss values
        loss_test = np.mean(loss_test_vector)

        # Saving the test loss of the current epoch for later plot
        loss_epoch_test.append(loss_test)

        # Saving the R2 scores of the current epoch for later plot
        R2_test = np.mean(R2_test)
        R2_epoch_test.append(R2_test)

        # Visualizing loss values against the number of epochs
        if (epoch+1)%5 == 0 and epoch != 0:
            visualization(loss_epoch_test, loss_epoch_train, R2_epoch_train, R2_epoch_test)
            plt.savefig('./checkpoints/visualization_plot.png', bbox_inches='tight') # saving plot

        # Saving the loss and the R2 score in a file
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
    
#------- VALIDATION PROCESS -------#

    # Defining the path from which it is possible to retrieve the best trained model
    PATH = './checkpoints/best_model.pt' 

    # Defining the model
    if params.masking==True:
        best_model = my_FNN_increasing_masking(dim_feat,params.dtype)
    else:
        best_model = my_FNN_increasing_NOmasking(dim_feat,params.dtype)

    checkpoint = torch.load(PATH)

    # Importing final state of the best model obtained during the training process
    best_model.load_state_dict(checkpoint['model_state'])

    # Importing model and move it to GPU (if available)
    if(torch.cuda.is_available()): # for the case of device with local GPU
        best_model = best_model.cuda()

    # After importing the model, we just need to compute the prediction on validation data
    with torch.no_grad():
        # Computing the output on validation data.
        # Since we use torch.no_grad(), the operation is not added to the computation graph and the procedure is therefore less complex
        output_validation = best_model(X_val)

    # Clearing active figures     
    plt.clf()

    # Visualizing cloud of points to see if the network performs better than the empirical approximation (logarithmic function) so far used in the field
    # The following plot offers an overall view about the performance of the network. For a better analysis, please refer to R2 score 
    # and correlation plot saved in checkpoints folder
    cloud_of_points(output_validation.cpu().detach().numpy(), y_val.cpu().detach().numpy(), 
                    X_val[:,0].cpu().detach().numpy(), mean_train[0], std_train[0])
    plt.savefig('./checkpoints/cloud_of_points.png', bbox_inches='tight')

    plt.clf()
    # Visualizing predicted values against theoretical ones
    # Depending on the total number of datapoints in validation set, it might be better to choose correlation_plot or correlation_plot_hist
    # Please refer to plots.py to know more about the difference
    correlation_plot(output_validation.cpu().detach().numpy(), y_val.cpu().detach().numpy())
    plt.savefig('./checkpoints/correlation_plot.png', bbox_inches='tight') # saving correlation plot
    correlation_plot_hist(output_validation.cpu().detach().numpy(), y_val.cpu().detach().numpy())
    plt.savefig('./checkpoints/correlation_plot_hist.png', bbox_inches='tight') # saving correlation plot hist









