""" 
This file is contains the neural network classes. 
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F

class TuneNet(nn.Module):
    """
    This model is used to tune the network architecture.
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

class PulsarDetectionNet(nn.Module):
    """
    This model is used to train on pulsar data.
    """
    def __init__(self, dim, num_classes, filters=None, bias=None):
        super(PulsarDetectionNet, self).__init__()

        if filters is not None:
            self.conv1 = nn.Conv2d(1, len(filters), kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(len(filters), 64, kernel_size=3, padding=1)
            self.conv1.weight = nn.Parameter(filters)

        if bias is not None:
            self.conv1.bias = nn.Parameter(bias)
            
        else:
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) 
            self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        
        # this part is to determine the shape of the first fully connected layer
        test_sample = torch.randn(dim, dim, dtype=torch.float32).view(-1,1,dim,dim)
        self._to_linear = None
        self.convs(test_sample)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
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

class SparseAutoencoder(nn.Module):
    def __init__(self, latent_size=32, criterion=None, sparsity_param=0.05, beta=1e-3):
        super(SparseAutoencoder, self).__init__()

        self.sparsity_param = sparsity_param
        self.beta = beta
        if criterion is None:
            self.criterion = nn.MSELoss(reduction='mean')
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bottleneck layer
        self.latent = nn.Conv2d(16, latent_size, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 16, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Sigmoid to map output to [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        self.encoded = self.latent(x)
        x = self.decoder(self.encoded)
        return x
    
    def kl_divergence(self, p, q):
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
    
    def loss_function(self, recon_x, x):
        # Reconstruction loss
        loss = self.criterion(recon_x, x)
        
        # Sparsity loss
        rho_hat = torch.mean(torch.sigmoid(self.encoded), dim=[0, 2, 3])
        sparsity_loss = torch.sum(self.kl_divergence(self.sparsity_param, rho_hat))
        
        return loss + self.beta * sparsity_loss