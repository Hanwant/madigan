import torch.nn as nn
import torch

from .common import NoisyLinear
from .utils import orthogonal_initialization


class QNetworkBaseMeta(type):
    """
    Useful for making sure that classes with this metaclass
    perform certain functions after being initialized
    I.e registering noisy linear layers or doing specific
    initialization
    """
    def __call__(cls, *args, **kw):
        instance = super().__call__(*args, **kw)
        instance.register_noisy_layers()
        print('registered noisy layers: ', instance.noisy_layers)
        instance.apply(orthogonal_initialization)
        print('done orthogonal initialization')
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
            elif isinstance(module, QNetworkBase) and module is not self:
                curr = type(self)
                mod = type(module)
                msg = f"Current Class ({curr}) inherits from QNetworkBase " + \
                    f"but wraps module ({mod}) also inheriting from " + \
                    "QNetworkBase. Ensure only one class inherits from " + \
                    "QNB so NoisyLinear layers only get registered once"
                raise TypeError(msg)
