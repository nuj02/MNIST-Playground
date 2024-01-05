import torch

class Autoencoder(torch.nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.conv1 = torch.nn.Conv1d()