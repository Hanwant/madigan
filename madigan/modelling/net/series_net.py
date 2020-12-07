"""

Pytorch implementation of SeriesNet, as described and implemented at:
https://github.com/kristpapadopoulos/seriesnet
Hyperparameter choices, are guided by the paper:
 SeriesNet: A Dilated Causal Convolution Neural Network for Forecasting
 Krist Papadopoulos, 2018
"""
from typing import List, Tuple
from functools import partial

import torch
import torch.nn as nn
from .common import PortEmbed, NormalHeadDQN, DuelingHeadDQN, NoisyLinear
from ...utils.data import State


class DilConvLayer(nn.Module):
    """
    1D Causal Convolution using small dilated kernels - For use in SeriesNet
    """
    def __init__(self, channels_in, channels_dim, kernel, dilation, dropout):
        super().__init__()
        self.act_fn = nn.GELU()
        self.conv_project = nn.Conv1d(channels_in,
                                      channels_dim,
                                      kernel,
                                      dilation=dilation,
                                      bias=False)
        padding = (dilation * (kernel - 1), 0)
        self.causal_padding_layer = nn.ReplicationPad1d(padding)
        self.conv_embed = nn.Sequential(self.causal_padding_layer,
                                        self.conv_project, self.act_fn)
        self.conv_compress = nn.Conv1d(channels_dim, 1, 1, bias=False)
        self.skip_conv = nn.Conv1d(channels_dim, 1, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        latent = self.conv_embed(x)
        res = x + self.conv_compress(latent)
        skip_out = self.dropout(self.skip_conv(latent))
        return res, skip_out


class SeriesNetQ(nn.Module):
    """
    Stacked Layers of 1D Dilated Convolutions.
    Dilations should increase by factor of 2 at each layer
    to grow receptive field while minimizing # parameters.
    Similar to WaveNet and Augmented WaveNet.
    See https://github.com/kristpapadopoulos/seriesnet/blob/master/seriesnet-Krist-Papadopoulos-v1.pdf
    """
    def __init__(self, input_shape: Tuple, output_shape: Tuple, d_model: int,
                 channel_dims: List, kernels: List, dilations: List,
                 dropouts: List, dueling: bool,
                 noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5,
                 **extra):
        super().__init__()
        if not all([
                len(channel_dims) == len(hp)
                for hp in (kernels, dilations, dropouts)
        ]):
            raise ValueError("Length for each of kernels, dilation, dropout" +
                             " must be all equal - corresponding to n_layers")
        input_dim = input_shape[1]
        input_length = input_shape[0]
        n_layers = len(channel_dims)
        self.layers = [
            DilConvLayer(input_dim, channel_dims[0], kernels[0], dilations[0],
                         dropouts[0])
        ]
        for i in range(1, n_layers):
            self.layers.append(
                DilConvLayer(1, channel_dims[i], kernels[i], dilations[i],
                             dropouts[i]))
        self.layers = nn.ModuleList(self.layers)
        n_assets = output_shape[0]
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        self.port_embed = Linear(n_assets + 1, d_model)
        self.conv_output_embed = Linear(input_length, d_model)
        if dueling:
            self.output_head = DuelingHeadDQN(d_model, output_shape, noisy_net,
                                           noisy_net_sigma)
        else:
            self.output_head = NormalHeadDQN(d_model, output_shape, noisy_net,
                                          noisy_net_sigma)
        self.output_act_fn = nn.GELU()
        self.apply(self.init)

    @staticmethod
    @torch.no_grad()
    def init(module):
        """
        Truncated Normal Initialization
        """
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.normal_(module.weight, mean=0, std=0.05)

    def get_state_emb(self, state: State):
        price = state.price.transpose(-1, -2)
        port = state.portfolio
        x, out = self.layers[0](price)
        for layer in self.layers[1:]:
            x, skip_out = layer(x)
            out += skip_out
        out = out.view(out.shape[0], -1)
        price_emb = self.conv_output_embed(self.output_act_fn(out))
        port_emb = self.port_embed(port)
        state_emb = price_emb * port_emb
        return state_emb

    def forward(self, state: State = None, state_emb: torch.Tensor = None):
        assert state is not None or state_emb is not None
        if state_emb is None:
            state_emb = self.get_state_emb(state)  # (bs, d_model)
        qvals = self.output_head(self.output_act_fn(state_emb))
        return qvals
