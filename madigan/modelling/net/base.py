import torch.nn as nn
import torch

from .common import NoisyLinear

class QNetworkBaseMeta(type):
    def __call__(cls, *args, **kw):
        instance = super().__call__(*args, **kw)
        instance.register_noisy_layers()
        print('registered noisy layers: ', instance.noisy_layers)
        return instance

class QNetworkBase(nn.Module, metaclass=QNetworkBaseMeta):
    def __init__(self):
        super().__init__()
        self.noisy_layers = None

    def sample_noise(self):
        """
        Samples noise for registered NoisyLinear layers.
        The noisy layers do an internal check for self.training
        So will not sample noise if model is in eval mode
        """
        if self.noisy_net:
            for module in self.noisy_layers:
                module.sample()

    def register_noisy_layers(self):
        """
        Call at end of init by the metaclass, to register noisy linear layers
        This is to make sample_noise() more efficient so it doesn't have to
        check all modules.

        """
        self.noisy_layers = []
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                self.noisy_layers.append(module)

