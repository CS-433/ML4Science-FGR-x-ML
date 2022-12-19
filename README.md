# ASTROLAB
In this repository you can find the code we used for the project proposed by [ASTROLAB](https://www.epfl.ch/labs/lastro/) at EPFL in collaboration with the global project [HIRAX](https://hirax.ukzn.ac.za/). The goal of this project was to train a model able to predict the Mass of Hydrogen in galaxies using astrophysical and cosmological input features, such as the mass of hosting halo ($M_{Halo}$), mass of black holes in the halo ($Mass_{BH}$), temperature (Temp), Density and SFR (Star Formation Rate)-  
In order to solve this problem, we trained a Fully-Connected Neural Network using `pytorch`. After optimizing the architecture using `talos` library, we ended up with two different networks depending on whether we focused on the whole dataset or only on halos having mass values larger then $10^10$ solar mass units.

The supervisor of the project was Dr. Michele Bianco.

## Team:
Our team (named `FGRxML`) is composed by:  
- Brioschi Riccardo: [@RiccardoBrioschi](https://github.com/RiccardoBrioschi)  
- D'Angeli Gabriele: [@gabrieledangeli](https://github.com/gabrieledangeli)  
- Di Gennaro Federico: [@FedericoDiGenanro](https://github.com/FedericoDiGennaro)  

## Environment:
We worked with `python3.8.5`. The Python libraries used are `numpy`,`pytorch1.13.0`,`talos`, `pandas`, `matplotlib` and `seaborn`.
Notice that, since we used `pytorch-cuda11.6`, the content of `main.py` and `talos_optimization.py` can be run using GPU if available.

## Data and reproducibility of the code
In order to reproduce the resultes showed in the paper, data (https://mega.nz/file/U1FTyALK#zr1NLKa_bEX9t3oFPTlYaw4sonbTuRVyWUXNsUcVQFk) must be placed in a folder called `outputs_test2` that has to be in the same working directory as the notebooks of this repo. `outputs_test2` will then contain several folders ($LH_{i}$) corresponding to simulations obtained considering different cosmological and astrophysical constants. 

## Description of notebooks
Here you can find what each file in the repo does. The order in which they are described follows the pipeline we used to obtain our results.
- `helpers.py`: implementation of  all the "support" functions used in others .py files.
- `neural_network.py`: implementation of the architecture we use to make our predictions.
- `params.py`:  here there are all the parameters we needed to train our model. We decided to create this file to increase the readability of the code.
- `main.py`: in this python file we actually train our model and generate all the plots we used in support of our numerical results.
- `talos_optimization.py`: in this file we used e `talos` library to do optimization of the following hyperparameters in our architecture: *number of layers, layer_size of first layer (the following layers grow as a power of 2), dropout, learning rate*.  


