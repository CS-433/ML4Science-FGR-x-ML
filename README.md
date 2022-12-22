# ASTROLAB
In this repository you can find the code we used for the project proposed by [ASTROLAB](https://www.epfl.ch/labs/lastro/) at EPFL in collaboration with the global project [HIRAX](https://hirax.ukzn.ac.za/). The goal of this project was to train a model able to predict the Mass of Hydrogen ( $M_{HI}$ ) in galaxies using astrophysical and cosmological input features, such as the mass of hosting halo ( $M_{Halo}$ ), mass of black holes in the halo ( $Mass_{BH}$ ), temperature (Temp), Density and SFR (Star Formation Rate).
In order to solve this problem, we trained a Fully-Connected Neural Network using `pytorch`. After optimizing the architecture using `talos` library, we ended up with two different networks depending on whether we focused on the whole dataset or only on halos having mass values larger than $10^{10}$ solar mass units.

The supervisor of the project was Dr. Michele Bianco.

## Team:
Our team (named `FGRxML`) is composed by:  
- Brioschi Riccardo: [@RiccardoBrioschi](https://github.com/RiccardoBrioschi)  
- D'Angeli Gabriele: [@gabrieledangeli](https://github.com/gabrieledangeli)  
- Di Gennaro Federico: [@FedericoDiGenanro](https://github.com/FedericoDiGennaro)  

## Environment:
We worked with `python3.8.5`. The Python libraries we used are `numpy`,`pytorch1.13.0`,`talos`, `pandas`, `matplotlib` and `seaborn`.
Notice that, since we used `pytorch-cuda11.6`, the content of `main.py` and `talos_optimization.py` can be run using GPU if available.

## Data and reproducibility of the code
In order to reproduce the resultes showed in the paper, [data](https://mega.nz/file/U1FTyALK#zr1NLKa_bEX9t3oFPTlYaw4sonbTuRVyWUXNsUcVQFk) must be placed in a folder called `outputs_test2` that has to be in the same working directory as the files of this repo. `outputs_test2` will then contain several folders ( $LH_{i}$ ) corresponding to simulations obtained considering different cosmological and astrophysical constants. As a last step, you must ensure to have txt files `params_IllustrisTNG.txt` (it is in the repo) and `redshifts.txt` (it is directly downloaded from the link provided) in the `outputs_test2` folder. Please notice that the `outputs_test2` you must take to run the code is the inner one that is downloaded from the link provided.  
After setting the environment correctly, you simply need to run the file `main.py`: as output, it will give you a `checkpoints` folder containing all the plots and the txt files with the results.  
Another important aspect of our code is that it enables to stop the training process and then to restart it from the point on which it was interrupted; thanks to this strategy, the network can be trained locally without being forced to wait until a very long training is completed, and at the same time ensuring a backup in case something goes wrong. When running infact, several files will be created and used to keep track of of all the provisional results. In order to do so, you only have to set the variable `first_run=False` in `params.py` without modifying the `checkpoints` folder that has been created in the previous run.

## Description of files
Here you can find a detailed description of what each file in this repository contains.
- `EXPLORATORY_ANALYSIS.ipynb`: jupyter notebook containing an initial exploratory analysis of the data computed at the beginning of the project. This analysis has been extremely useful in order to identify informative features and remove useless covariates.
- `helpers.py`: implementation of all the "support" functions used in others .py files.
- `main.py`:  python file used to train and validate the final model and generate all the plots used as support for the numerical results we obtained.
- `neural_network.py`: implementation of the architectures we used to obtain final results and predictions.
- `params.py`: file containing the parameters we had to set before training the final model. We decided to create this file to increase the readability of the code.
- `plots.py`: file containing functions used to plot results and save images, later stored in `results` folder.
- `results_talos.ipynb` : jupyter notebook containing a detailed analysis of the results obtained after running `talos_optimization.py`. This notebook was used in order to choose the best set of hyperparameters to use when running `main.py`.
- `talos_optimization.py`: in this file we used  `talos` library to find the best hyperparameters for the final architecture: *number of layers, layer_size of first layer (the following layers grow as a power of 2), dropout, learning rate*. 
- `utils.cosm.py`: file containing helper functions used to approximate data using state-of-the-art approximating models (e.g; recent model suggested by Padmanabhan). 
- `report.pdf`: final report of our project

## Description of folders
Here you can find a detailed description of what each folder in this repository contains.
- `results`: folder containing the results (loss values, $R^2$ score, images) related to the final models presented in the paper.
- `talos_results`: folder containing the results produced by `talos_optimization.py`.
- `WEEKLY MEETING`: folder containing presentations documenting the results and improvements obtained while working on the project.




