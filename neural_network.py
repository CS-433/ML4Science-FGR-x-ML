" File to define model"

import torch.nn as nn
import torch.nn.functional as func


##### FIRST ARCHITECTURE #####

class my_FNN_increasing(nn.Module):

    " Class to define the first architecture"

    def __init__(self,num_feature, dtype):
        # Importing methods and attributes from Module
        super().__init__()
        # Generating and initializing each layer
        self.l1 = nn.Linear(num_feature,64,dtype=dtype)
        self.reLU1 = nn.ReLU()
        self.l2 = nn.Linear(64,128,dtype=dtype)
        self.reLU2 = nn.ReLU()
        self.l3 = nn.Linear(128,256,dtype=dtype)
        self.reLU3 = nn.ReLU()
        self.l4 = nn.Linear(256,1,dtype=dtype)
        #self.dropout = nn.Dropout(0.25)

    def forward(self,input):

        "Function to implement the forward pass"

        out = self.l1(input)
        out = self.reLU1(out)
        #out = self.dropout(out)
        out = self.l2(out)
        out = self.reLU2(out)
        #out = self.dropout(out)
        out = self.l3(out)
        out = self.reLU3(out)
        #out = self.dropout(out)
        out = self.l4(out)
        #out = self.l5(out)
        
        return out

##### SECOND ARCHITECTURE #####
    
class my_FNN_mirror(nn.Module):

    " Class to define the second architecture"

    def __init__(self,num_feature,dtype):
        # Importing methods and attributes from Module
        super().__init__()
        # Generating and initializing each layer
        self.l1 = nn.Linear(num_feature,64,dtype=dtype)
        self.reLU1 = nn.LeakyReLU()
        self.l2 = nn.Linear(64,128,dtype=dtype)
        self.reLU2 = nn.LeakyReLU()
        self.l3 = nn.Linear(128,64,dtype=dtype)
        self.reLU3 = nn.LeakyReLU()
        self.l4 = nn.Linear(64,1,dtype=dtype)

        #self.dropout = nn.Dropout(0.25)

    def forward(self,input):

        "Function to implement the forward pass"

        out = self.l1(input)
        out = self.reLU1(out)
        #out = self.dropout(out)
        out = self.l2(out)
        out = self.reLU2(out)
        #out = self.dropout(out)
        out = self.l3(out)
        out = self.reLU3(out)
        #out = self.dropout(out)
        out = self.l4(out)

        return out


