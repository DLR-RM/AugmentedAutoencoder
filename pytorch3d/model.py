import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.l1 = nn.Linear(128,128)
        self.l2 = nn.Linear(128,64)
        self.l3 = nn.Linear(64,3)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.tanh = nn.Tanh()


    # Input: x = lantent code
    # Output: y = pose as quaternion
    def forward(self,x):
        #x = F.relu(self.bn1(self.l1(x)))
        #x = F.relu(self.bn2(self.l2(x)))
        #x = F.relu(self.l1(x))
        #x = F.relu(self.l2(x))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        #x = self.l1(x)
        #x = self.l2(x)
        
        y = self.tanh(self.l3(x))
        #y = self.l3(x)
        return y
