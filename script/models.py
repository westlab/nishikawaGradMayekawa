import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, height, width, mid_dim=64, z_dim=32):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(height * width, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, z_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, height * width),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x