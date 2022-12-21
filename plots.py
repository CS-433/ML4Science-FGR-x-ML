""" Function to visualize results and compare them to theoretical values.
    The plots presentend in the final paper have been produced using the following functions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

##### VISUALIZATION TOOLS #####


def visualization(losses_test, losses_train, R2_train, R2_test):
    """
    Visualization of test loss (y axis) versus number of epochs (x axis)

    Args:
        losses_test: array of shape (num_epochs,) containing test error
        losses_train: array of shape (num_epochs,) containing train error
    """
    fig,axs = plt.subplots(1,2, figsize=(15,10))
    axs[0].plot(range(1,len(losses_train)+1), losses_train, 'bo-', label='Loss(MSE) train')
    axs[0].plot(range(1,len(losses_test)+1), losses_test, 'ro-', label='Loss(MSE) test')
    axs[0].set(title='MSE w.r.t. number of epochs',xlabel='epochs',ylabel='test_loss(MSE)')
    axs[0].grid(visible=True)
    # Setting log scale to better visualize the results
    axs[0].set_yscale('log')
    axs[0].legend()

    axs[1].plot(range(1,len(R2_train)+1), R2_train, 'bo-', label='R2 score train')
    axs[1].plot(range(1,len(R2_test)+1), R2_test, 'ro-', label='R2 score test')
    axs[1].set(title='R2 score w.r.t. number of epochs',xlabel='epochs',ylabel='R2 score')
    axs[1].grid(visible=True)
    axs[1].set(ylim = [-1,1])
    axs[1].legend()


def correlation_plot_hist(predicted, y):
    """
    Visualization of predicted values against theoretical values using a logarithmic scale on both axes. 
    The following function returns a 2d histogram to better visualize the result despite the large number of overlapping datapoints
    
    Args:
        predicted: array of shape (N,) containing values predicted using neural network
        y: array of shape (N,) containing target values
    """
    # Converting output values to original scale
    predicted = 10**predicted -1
    y = 10**y - 1
    # Defining bins on both axes
    N, xbin, ybin = np.histogram2d(y, predicted, bins=200)
    N = N.T
    # Centering the bins
    x_cent = 0.5*(xbin[1:]+xbin[:-1])
    y_cent = 0.5*(ybin[1:]+ybin[:-1])

    # Plotting the final result showing a colormap that reflects the density of points in every bin
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(131, title='Correlation plot')
    plt.pcolormesh(x_cent, y_cent, N, norm=LogNorm())
    plt.colorbar()

def correlation_plot(predicted, y):
    """
    Visualization of predicted values against theoretical values using a logarithmic scale on both axes.
    The following function returns a scatter plot of the data. Depending on the total number of the datapoints, it might be better
    to choose correlation_plot_hist
    
    Args:
        predicted: array of shape (N,) containing values predicted using neural network
        y: array of shape (N,) containing target values
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    # Plotting predicted and theoretical values after rescaling them to original scale
    ax.scatter(10**y - 1, 10**predicted -1, edgecolors=(0, 0, 0))
    ax.plot([min(10**y -1), max(10**y -1)], [min(10**y -1), max(10**y -1)], 'r--', lw=4)

    # Adding lines representing a confidence interval of width = 2*sigma (around 68 % confidence interval)
    ax.plot([min(10**y -1), max(10**y -1)], [min(10**y -1)*(1+0.34), max(10**y -1)*(1+0.34)], 'y--', lw=2)
    ax.plot([min(10**y -1), max(10**y -1)], [min(10**y -1)*(1-0.34), max(10**y -1)*(1-0.34)], 'y--', lw=2)
    
    ax.set(xlabel='Original', ylabel='Predicted', xscale='log', yscale='log')
    ax.set_title('Correlation plot: True values vs Predicted values')

def cloud_of_points(predictions,target,massHalo, mean_halo, std_halo):
    """
    Function to plot MHI against massHalo. This function returns a comparison between the predicted and the true cloud of points,
    giving an overall view of the performance of the model. To further investigate the quality of every single prediction computed by the
    neural network, it might be better to use correlation_plot or correlation_plot_hist depending on the total number of datapoints
    
    Args:
        predictions: array of shape (N,) containing values predicted using neural network
        target: array of shape (N,) containing target values
        massHalo: array of shape (N,) containing MassHalo values
        meanHalo : scalar corresponding to the mean of MassHalo (the value is computed during the standardization of input train data)
        stdHalo: scalar corresponding to the standard deviation of MassHalo (the value is computed during the standardization of input train data)
    """

    # Converting massHalo values to original scale
    massHalo = 10**((massHalo*std_halo) + mean_halo) - 1

    # Converting output values to original scale
    predictions = 10 ** predictions -1
    target = 10 ** target -1

    #Plotting theoretical and predicted results in order to compare them 
    fig, axs = plt.subplots(1,2, figsize = (10,5))
    axs[0].scatter(massHalo, predictions, alpha = 0.8, marker = '.')
    axs[0].set_title('Scatter plot using predicted data')
    axs[0].set(xscale='log', yscale='log', xlim=(1e7, 1e14), ylim=(1e-2, 1e12), xlabel='MassHalo', ylabel='MassHI')

    axs[1].scatter(massHalo,target, alpha = 0.8, marker = '.')
    axs[1].set_title('Scatter plot using original data')
    axs[1].set(xscale='log', yscale='log', xlim=(1e7, 1e14), ylim=(1e-2, 1e12), xlabel='MassHalo', ylabel='MassHI' )
