from functools import reduce
from collections.abc import Iterable

import torch
import torch.nn as nn

from .base import QNetwork
from ...utils.data import State


class PortEmbed(nn.Module):
    """
    Create embedding from portfolio
    """
    def __init__(self, n_assets, d_model):
        super().__init__()
        self.embed = nn.Linear(n_assets, d_model)
    def forward(self, raw_port):
        return self.embed(raw_port)

class NormalHead(nn.Module):
    def __init__(self, d_model, output_shape):
        super().__init__()
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        self.out = nn.Linear(d_model, self.n_assets*self.action_atoms)
    def forward(self, state_emb):
        qvals = self.out(state_emb).view(state_emb.shape[0],
                                         self.n_assets, self.action_atoms)
        return qvals

class DuelingHead(nn.Module):
    def __init__(self, d_model, output_shape):
        super().__init__()
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        self.value_net = nn.Linear(d_model, self.n_assets)
        self.adv_net = nn.Linear(d_model, self.n_assets*self.action_atoms)
    def forward(self, state_emb):
        value = self.value_net(state_emb)
        adv = self.adv_net(state_emb).view(state_emb.shape[0],
                                           self.n_assets, self.action_atoms)
        qvals = value[..., None] + adv - adv.mean(-1, keepdim=True)
        return qvals

class GRUNet(nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple, n_layers=3,
                 d_model=128, d_model_project=256, dueling=True, **extra):
        """
        input_shape: (window_length, n_features)
        output_shape: (n_assets, action_atoms)
        """
        super().__init__()

        assert len(input_shape) == 2
        window_len = input_shape[0]
        n_features = input_shape[1]
        self.input_shape = input_shape
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        self.d_model = d_model

        self.layers = nn.GRU(n_features, d_model, n_layers, batch_first=True)
        self.act = nn.GELU()
        self.price_project = nn.Linear(d_model, d_model_project)
        self.port_project = nn.Linear(self.n_assets, d_model_project)
        if dueling:
            self.output_head = DuelingHead(d_model, output_shape)
        else:
            self.output_head = NormalHead(d_model, output_shape)

    def get_state_emb(self, state: State):
        """
        Given a State object containing tensors as .price and .portfolio
        attributes, returns an embedding of shape (bs, d_model)
        """
        price = state.price.transpose(-1, -2) # switch features and time dimension
        port = state.portfolio
        price_emb = self.layers(price).view(price.shape[0], -1)[:, -1]
        import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        price_emb = self.price_project(price_emb)
        port_emb = self.port_project(port)
        state_emb = price_emb * port_emb
        out = self.act(state_emb)
        return out

    def forward(self, state: State = None, state_emb: torch.Tensor = None):
        """
        Returns qvals given either state or state_emb
        output_shape = (bs, n_assets, action_atoms)
        """
        assert state is not None or state_emb is not None
        if state_emb is None:
            state_emb = self.get_state_emb(state) # (bs, d_model)
        qvals = self.output_head(state_emb) # (bs, n_assets, action_atoms)
        return qvals
