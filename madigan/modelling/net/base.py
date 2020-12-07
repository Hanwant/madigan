import torch.nn as nn
import torch

from .common import NoisyLinear

class QNetworkBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.noisy_layers = None

    def sample_noise(self):
        if self.noisy_net:
            for module in self.noisy_layers:
                module.sample()

    def register_noisy_layers(self):
        self.noisy_layers = []
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                self.noisy_layers.append(module)

