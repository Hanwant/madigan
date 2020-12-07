from functools import partial

import torch.nn as nn
from .base import QNetworkBase
from .common import NoisyLinear


class MLPNet(QNetworkBase):
    def __init__(self,
                 in_shape: tuple,
                 out_shape: tuple,
                 d_model=256,
                 n_layers=2,
                 act=nn.ReLU,
                 noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5,
                 **params):
        super().__init__()
        assert len(in_shape) == 1, "in_shape should be a size 1 tuple/list"
        assert len(out_shape) == 2, "out_shape should be size 2"
        self.in_shape = in_shape[0]
        self.out_shape = out_shape
        self.out_len = out_shape[0] * out_shape[1]
        self.d_model = d_model
        self.n_layers = n_layers
        self.act = act
        self.in_head = nn.Linear(self.in_shape, d_model)
        layers = []
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        for i in range(n_layers):
            layers.append(Linear(d_model, d_model))
            layers.append(self.act())
        self.layers = nn.Sequential(*layers)
        self.out_head = Linear(d_model, self.out_len)

    def forward(self, state=None, state_emb=None):
        assert state is not None or state_emb is not None
        if state_emb is None:
            state_emb = self.get_state_emb(state)
        return self.out_head(state_emb).view(state_emb.shape[0],
                                             *self.out_shape)

    def get_state_emb(self, state):
        return self.layers(self.in_head(state))
