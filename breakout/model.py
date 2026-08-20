"""Rete convoluzionale DQN per input Atari 4x84x84."""

import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(states.float() / 255.0))
