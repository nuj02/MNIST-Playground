import torch.nn as nn

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=1000),
            nn.BatchNorm1d(num_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=500),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=250),
            nn.BatchNorm1d(num_features=250),
            nn.ReLU(),
            nn.Linear(in_features=250,out_features=2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=2, out_features=250),
            nn.BatchNorm1d(num_features=250),
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=500),
            nn.BatchNorm1d(num_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=1000),
            nn.BatchNorm1d(num_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000,out_features=784),
            nn.Sigmoid(),
            nn.Unflatten(dim=1,unflattened_size=(1,28,28))
        )

    def forward(self, input):
        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return encoded, decoded