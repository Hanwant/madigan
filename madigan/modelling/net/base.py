import torch.nn as nn
import torch


class QNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape
        self.output_shape

    def forward(self, state=None, state_emb=None):
        raise NotImplementedError

    def get_state_emb(self, state):
        raise NotImplementedError

