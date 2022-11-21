" File to define model"

import torch.nn as nn
import torch.nn.functional as func


##### FIRST ARCHITECTURE #####

class my_FNN_increasing(nn.Module):

    " Class to define the first architecture"

    def __init__(self,num_feature):
        # Importing methods and attributes from Module
        super().__init__()
        # Generating and initializing each layer
        self.l1 = nn.Linear(num_feature,64)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(64,128)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(128,256)
        self.l6 = nn.Sigmoid()
        self.l7 = nn.Linear(256,1)

    def forward(self,input):

        "Function to implement the forward pass"

        out = self.l1(input)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)

        return out

##### SECOND ARCHITECTURE #####
    
class my_FNN_mirror(nn.Module):

    " Class to define the second architecture"

    def __init__(self,num_feature):
        # IMporting methods and attributes from Module
        super().__init__()
        # Generating and initializing each layer
        self.l1 = nn.Linear(num_feature,64)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(64,128)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(128,64)
        self.l6 = nn.Sigmoid()
        self.l7 = nn.Linear(64,1)

    def forward(self,input):

        "Function to implement the forward pass"

        out = self.l1(input)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)

        return out


