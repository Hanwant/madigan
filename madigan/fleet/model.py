import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, state=None, state_emb=None):
        raise NotImplementedError

    def get_state_emb(self, state):
        raise NotImplementedError
