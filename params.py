" Useful variables to define model"

import torch

# The following parameters are used in order to define the architecture in the file main.py. They are moreover essential in order to run the code using EPFL cluster.

first_run= True # set to True if the current run is the first one. If True, checkpoints folder is created.

batch_size = 256 # batch size used in the training and test process. Considering the large amount of data, 512 is another possible value.

lr = 0.01 # learning rate given as input to the optimizer. During the learning process, this value might be adjusted by the scheduler.

epochs = 10 # number of total epochs used during the training process.

dtype = torch.float64 # data type used to initialize each layer of the model. torch.float32 cannot be used because of the large values contained in the dataset.


