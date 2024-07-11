""" 
This file is contains the neural network classes. 
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F

class TuneNet(nn.Module):
    """
    This module is used to tune the network architecture.
    """

    def __init__(self, dim, c1=16, c2=32, c3=None, fc=64, f_size=3):
        super(TuneNet, self).__init__()
        self.conv1 = nn.Conv2d(1, c1, kernel_size=f_size)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=f_size)
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=f_size) if c3 is not None else None

        test_sample = torch.randn(dim, dim).view(-1,1,dim,dim)
        self._to_linear = None
        self.convs(test_sample)

        self.fc1 = nn.Linear(self._to_linear, fc)
        self.fc2 = nn.Linear(fc, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        if self.conv3 is not None:
            x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        
        return x

class PulsarDetectionCNN_1(nn.Module):
    def __init__(self, dim, filters=None, bias=None):
        super(PulsarDetectionCNN_1, self).__init__()

        if filters is not None:
            self.conv1 = nn.Conv2d(1, len(filters), kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(len(filters), 32, kernel_size=3, padding=1)
            self.conv1.weight = nn.Parameter(filters)

        if bias is not None:
            self.conv1.bias = nn.Parameter(bias)
            
        else:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) 
            self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        
        # this part is to determine the shape of the first fully connected layer
        test_sample = torch.randn(dim, dim).view(-1,1,dim,dim)
        self._to_linear = None
        self.convs(test_sample)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 2)
    
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        
        return x