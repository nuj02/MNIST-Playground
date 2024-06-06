import torch
import torch.nn as nn
# import torch.nn.functional as F

class Autoencoder(torch.nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.flatten = nn.Flatten()

        self.encode1 = nn.Linear(in_features=784,
                                 out_features=1000)
        
        self.encode2 = nn.Linear(in_features=1000,
                                 out_features=500)
        
        self.encode3 = nn.Linear(in_features=500,
                                 out_features=250)
        
        self.encode4 = nn.Linear(in_features=250,
                                 out_features=2)
        
        self.decode4 = nn.Linear(in_features=2,
                                 out_features=250)
        
        self.decode3 = nn.Linear(in_features=250,
                                 out_features=500)
        
        self.decode2 = nn.Linear(in_features=500,
                                 out_features=1000)
        
        self.decode1 = nn.Linear(in_features=1000,
                                 out_features=784)
        
        self.unflatten = nn.Unflatten(1,(28,28))
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        x = self.encode4(x)
        x = self.decode4(x)
        x = self.decode3(x)
        x = self.decode2(x)
        x = self.decode1(x)
        x = self.unflatten(x)
        
        return x
    
    