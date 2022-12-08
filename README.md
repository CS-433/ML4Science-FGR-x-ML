# ASTROLAB
In the repository you can find the code we used for a project proposed by [ASTROLAB](https://www.epfl.ch/labs/lastro/) at EPFL in collaboration with the global project [HIRAX](https://hirax.ukzn.ac.za/). The goal of this project was to train a model able to predict the Mass of Hydrogen in galaxies based on some other features such as Mass of halos, mass of black holes in the halo, Temperature, Density, SFR (Star Formation Rate),...  
To solve this problem, we trained a Fully-Connected Neural Network using `pytorch`. After optmizing the architecture using `talos`,  we ended up with the following network:

The supervisor of the project was Dr. Michele Bianco.

## Team:
Our team (named `FGRxML`) is composed by:  
- Brioschi Riccardo: [@RiccardoBrioschi](https://github.com/RiccardoBrioschi)  
- D'Angeli Gabriele: [@gabrieledangeli](https://github.com/gabrieledangeli)  
- Di Gennaro Federico: [@FedericoDiGenanro](https://github.com/FedericoDiGennaro)  

# Project pipeline

## Environment:
We worked with `python3.8.5`. The Python's libraries used are `numpy`,`pytorch`,`talos`, `pandas`, `matplotlib` and `seaborn`.

## Data and reproducibility of the code
For the code reproducibility, the datasets must be placed in a folder called `?????????` that must be in the same working directory as the notebooks in this repo. The `data` folder must contain the other folders that refers to each LH called `LH_i`. In each folder LH_i, there have to be HDF5 files that we read in our code to build our data matrix $X$.

## Description of notebooks
Here you can find what each file in the repo does. The order in which they are described follows the pipeline we used to obtain our results.
- `helpers.py`: implementation of  all the "support" functions used in others .py files.
- `neural_network.py`: implementation of the architecture we use to make our predictions.
- `params.py`:  here there are all the parameters we needed to train our model. We decided to create this file to increase the readability of the code.
- `main.py`: in this python file we actually train our model and generate all the plots we used in support of our numerical results.
- `talos_optimization.py`: in this file we used e `talos` library to do optimization of the following hyperparameters in our architecture: *number of layers, layer_size of first layer (the following layers grow as a power of 2), dropout, learning rate*.  


