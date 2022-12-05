""" Function to visualize results and compare them to theoretical values"""

import matplotlib.pyplot as plt
import numpy as np


##### VISUALIZATION TOOLS #####


def visualization(losses_test, losses_train, R2_train, R2_test):
    """
    Visualization of test loss (y axis) versus number of epochs that refer to that loss (x axis)

    Args:
        losses_test: array of shape (num_epochs,) containing test error
        losses_train: array of shape (num_epochs,) containing train error
    """
    fig,axs = plt.subplots(1,2, figsize=(15,10))
    axs[0].plot(range(1,len(losses_train)+1), losses_train, 'bo-', label='Loss(MSE) train')
    axs[0].plot(range(1,len(losses_test)+1), losses_test, 'ro-', label='Loss(MSE) test')
    axs[0].set(title='MSE w.r.t. number of epochs',xlabel='epochs',ylabel='test_loss(MSE)')
    axs[0].grid(visible=True)
    axs[0].set_yscale('log')
    axs[0].legend()

    axs[1].plot(range(1,len(R2_train)+1), R2_train, 'bo-', label='R2 score train')
    axs[1].plot(range(1,len(R2_test)+1), R2_test, 'ro-', label='R2 score test')
    axs[1].set(title='R2 score w.r.t. number of epochs',xlabel='epochs',ylabel='R2 score')
    axs[1].grid(visible=True)
    axs[1].set(ylim = [-1,1])
    axs[1].legend()

def correlation_plot(predicted, y):
    """
    Visualization of redicted values against theretical values using a logarithmic scale on both axes
    
    Args:
        predicted: array of shape (N,) containing values predicted using neural network
        y: array of shape (N,) containing target values
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(10**y, 10**predicted, edgecolors=(0, 0, 0))
    ax.plot([min(10**y), max(10**y)], [min(10**y), max(10**y)], 'r--', lw=4)

    # Adding lines representing a confidence interval of width = 2*sigma
    ax.plot([min(10**y), max(10**y)], [min(10**y)*(1+0.34), max(10**y)*(1+0.34)], 'y--', lw=2)
    ax.plot([min(10**y), max(10**y)], [min(10**y)*(1-0.34), max(10**y)*(1-0.34)], 'y--', lw=2)
    
    ax.set(xlabel='Original', ylabel='Predicted', xscale='log', yscale='log')
    ax.set_title('Correlation plot: True values vs Predicted values')

def cloud_of_points(predictions,target,massHalo, mean_halo, std_halo):
    """
    Function to plot  MHI against massHalo
    
    Args:
        predictions: array of shape (N,) containing values predicted using neural network
        target: array of shape (N,) containing target values
        massHalo: array of shape (N,) containing MassHalo values
        meanHalo : scalar corresponding to the mean of MassHalo (the value is computed during the standardization of input data)
        stdHalo: scalar correspondng to the standard deviation of MassHalo (the value is computed during the standardization of input data)
    """

    # Converting massHalo values to original scale
    massHalo = (massHalo*std_halo) + mean_halo

    # Converting output values to original scale
    predictions = 10 ** predictions
    target = 10 ** target

    #Plotting theoretical and predicted result in order to compare them 
    fig, axs = plt.subplots(1,2, figsize = (10,5))
    axs[0].scatter(massHalo, predictions, alpha = 0.8, marker = '.')
    axs[0].set_title('Scatter plot using predicted data')
    axs[0].set(xscale='log', yscale='log', xlim=(1e7, 1e14), ylim=(1e-2, 1e12), xlabel='MassHalo', ylabel='MassHI')

    axs[1].scatter(massHalo,target, alpha = 0.8, marker = '.')
    axs[1].set_title('Scatter plot using original data')
    axs[1].set(xscale='log', yscale='log', xlim=(1e7, 1e14), ylim=(1e-2, 1e12), xlabel='MassHalo', ylabel='MassHI' )
