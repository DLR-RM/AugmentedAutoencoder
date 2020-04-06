import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelEncoder(nn.Module):
    def __init__(self, output_size=4):
        super(ModelEncoder, self).__init__()

        self.cn1 = nn.Conv2d(3, 128, (5,5), stride=2)
        self.cn2 = nn.Conv2d(128, 256, (5,5), stride=2)
        self.cn3 = nn.Conv2d(256, 512, (5,5), stride=2)
        self.cn4 = nn.Conv2d(512, 512, (5,5), stride=2)

        self.fc = nn.Linear(12800, 128)
        
        self.l1 = nn.Linear(128,128)
        self.l2 = nn.Linear(128,64)
        self.l3 = nn.Linear(64,output_size)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self,x):
        x = F.relu(self.cn1(x))
        x = F.relu(self.cn2(x))
        x = F.relu(self.cn3(x))
        x = F.relu(self.cn4(x))

        x = x.flatten(start_dim=1)
        x = self.fc(x)
        
        x = F.relu(self.bn1(self.l1(x)))
        x = F.relu(self.bn2(self.l2(x)))              
        y = self.l3(x)
        return y
