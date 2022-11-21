" File to define model"

import torch.nn as nn
import torch.nn.functional as func


class my_FNN_increasing(nn.Module):

    def __init__(self,batch_size,num_feature):

        self.l1 = nn.Linear(num_feature,64)
        self.l2 = nn.Sigmoid()
        self.l3 = nn.Linear(64,128)
        self.l4 = nn.Sigmoid()
        self.l5 = nn.Linear(128,256)
        self.l6 = nn.Sigmoid()
        self.l7 = nn.Linear(256,1)

    def forward(self,input):

        out = self.l1(input)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = self.l6(out)
        out = self.l7(out)

        return out

