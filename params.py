" Useful variables to define model"

import torch

# The following parameters are used in order to define the architecture in the file main.py. They are moreover essential in order to run the code using EPFL cluster.

# Please refer to neural_network.py to know how to set these parameters when running the main in order to obtain the results showed in the final
# paper

first_run= True # set to True if the current run is the first one. If True, checkpoints folder is created. If false, the previous model
                # saved in checkpoints folder is retrieved and the training process can continue. It is especially useful when training for 
                # a large number of epochs or when using an external cluster

batch_size = 512 # batch size used in the training and test process. Please refer to neural_network.py to set this parameter in a proper way
                 # depending on the architecture you are willing to use

lr = 0.01 # learning rate given as input to the optimizer. During the learning process, this value might be adjusted by the scheduler.
          # Please refer to neural_network.py to set this parameter in a proper way depending on the architecture you are willing to use

epochs = 1000 # number of total epochs used during the training process. When using the whole dataset (around 3 millions datapoints), it might 
              # be convenient to choose a low value in order to reduce the computationl complexity of the process

dtype = torch.float64 # data type used to initialize each layer of the model

masking = False # boolean indicating whether the masking procedure should be computed or not when retrieving data. This value depends on the
                # choice of the architecture
