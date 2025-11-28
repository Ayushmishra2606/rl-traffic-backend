import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )


    def forward(self, x):
        return self.net(x)