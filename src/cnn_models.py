""" 
This file is contains the neural network classes. 
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F

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
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # this part is to determine the shape of the first fully connected layer
        test_sample = torch.randn(dim, dim).view(-1,1,dim,dim)
        self._to_linear = None
        self.convs(test_sample)

        self.fc1 = nn.Linear(self._to_linear, 64)
        self.fc2 = nn.Linear(64, 2)
    
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