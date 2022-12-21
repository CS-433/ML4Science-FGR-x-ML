" File to define model"

import torch.nn as nn
import torch.nn.functional as func
import talos
import params


# Defining neural network architecture to use Talos optimization. 
# The results obtained doing cross validation are then used to build the neural network used in main script.
# You can find the final version of our neural networks at the end of this file

##### ARCHITECTURE USED TO PERFORM OPTIMIZATION #####

class customized_increasing_NN(nn.Module, talos.utils.TorchHistory):

    """ Class to run optimization script. The exact architecture (number of hidden layers, size and Dropout rate) depends on the
    dictionary passed as an input when creating the object."""

    def __init__ (self,p, num_features,dtype):
        # Importing methods and attributes from Module
        super().__init__()
        # We use linear layer, we use params to define other layers
        self.dropout = nn.Dropout(p['dropout'])
        # Defining activation function to use
        self.activation = p['activation']
        # Defining the structure of the first hidden layer
        self.starting_linear = nn.Linear(num_features, p['hidden_layer_size'], dtype=dtype)
        # Defining the total number of hidden layers in the network
        self.__nr_layers = p['nr_layers']
        # Defining the size of the initial layer of the network. As described in the related paper, the size of the following layers increases
        # by powers of two
        self.__hidden_layer_size = p['hidden_layer_size']
        # Defining output layer
        self.ending_layer = nn.Linear((2**(self.__nr_layers-1))*(self.__hidden_layer_size), 1, dtype=dtype)
    
    def forward(self,input):

        "Function to implement the forward pass"

        out = self.dropout(self.activation(self.starting_linear(input)))
        for idx_layer in range(0,self.__nr_layers-1):
            self.linear = nn.Linear(self.__hidden_layer_size*(2**idx_layer), self.__hidden_layer_size*(2**(idx_layer+1)), dtype = params.dtype).cuda()
            out = self.dropout(self.activation(self.linear(out)))
        out = self.ending_layer(out)
        return out

##### ARCHITECTURES USED TO RUN MAIN.PY AND OBTAIN FINAL RESULTS SHOWED IN THE PAPER #####

# The hyperparameters of these architectures (e.g. depth, num layers, activation function, dropout rate) have been decided after optimizing the model using talos library.
# For further details, please refer to talos_optimization

class my_FNN_increasing_masking(nn.Module):

    """Class to define the architecture. The hyperparameters chosen after talos optimization are:
    {'nr_layers:4, hidden_layer_size:16, activation: ReLU(), dropout_rate:0.05, lr:0.01}.
    Notice that nr_layers refers to the number of hidden layers without considering the one taking the initial datapoints as input. This choice
    of notation was made while implementing the optimization process.
    This architecture has been used on the whole dataset (every simulation) after applying the masking procedure.
    You need to set lr = 0.01 and batch_size = 256 in params before using this architecture.
    """

    def __init__(self,num_feature, dtype):
        # Importing methods and attributes from Module
        super().__init__()
        # Generating and initializing each layer
        self.l1 = nn.Linear(num_feature,16,dtype=dtype)
        self.reLU1 = nn.ReLU()
        self.l2 = nn.Linear(16,32,dtype=dtype)
        self.reLU2 = nn.ReLU()
        self.l3 = nn.Linear(32,64,dtype=dtype)
        self.reLU3 = nn.ReLU()
        self.l4 = nn.Linear(64,128,dtype=dtype) 
        self.reLU4 = nn.ReLU()
        self.l5 = nn.Linear(128,1,dtype=dtype) 
        self.dropout = nn.Dropout(0.05)

    def forward(self,input):

        "Function to implement the forward pass"

        out = self.l1(input)
        out = self.reLU1(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.reLU2(out)
        out = self.dropout(out)
        out = self.l3(out)
        out = self.reLU3(out)
        out = self.dropout(out)
        out = self.l4(out)
        out = self.reLU4(out)
        out = self.dropout(out)
        out = self.l5(out)
  
        return out

class my_FNN_increasing_NOmasking(nn.Module):

    """Class to define the architecture. The hyperparameters chosen after talos optimization are:
    {'nr_layers:4, hidden_layer_size:32, activation: LeakyReLU(), dropout_rate:0.1, lr:0.01}.
    Notice that nr_layers refers to the number of hidden layers without considering the one taking the initial datapoints as input. This choice
    of notation was made while implementing the optimization process.
    This architecture has been used on the whole dataset (every simulation) WITHOUT applying the masking procedure.
    You need to set lr = 0.01 and batch_size = 16000 in params before using this architecture.
    """

    def __init__(self,num_feature, dtype):
        # Importing methods and attributes from Module
        super().__init__()
        # Generating and initializing each layer
        self.l1 = nn.Linear(num_feature,32,dtype=dtype)
        self.reLU1 = nn.LeakyReLU() # notice that we work using the default slope for negative values of the input
        self.l2 = nn.Linear(32,64,dtype=dtype)
        self.reLU2 = nn.LeakyReLU()
        self.l3 = nn.Linear(64,128,dtype=dtype)
        self.reLU3 = nn.LeakyReLU()
        self.l4 = nn.Linear(128,256,dtype=dtype) 
        self.reLU4 = nn.LeakyReLU()
        self.l5 = nn.Linear(256,1,dtype=dtype) 
        self.dropout = nn.Dropout(0.1)

    def forward(self,input):

        "Function to implement the forward pass"

        out = self.l1(input)
        out = self.reLU1(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.reLU2(out)
        out = self.dropout(out)
        out = self.l3(out)
        out = self.reLU3(out)
        out = self.dropout(out)
        out = self.l4(out)
        out = self.reLU4(out)
        out = self.dropout(out)
        out = self.l5(out)
  
        return out
