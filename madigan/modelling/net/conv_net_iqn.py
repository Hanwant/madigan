import math
from functools import reduce, partial

import torch
import torch.nn as nn

from .common import NoisyLinear
from .base import QNetworkBase
from .utils import calc_conv_out_shape, calc_pad_to_conserve
from ...utils.data import State


class NormalHeadIQN(nn.Module):
    def __init__(self,
                 d_model: int,
                 output_shape: tuple,
                 noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5):
        super().__init__()
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        self.out = Linear(d_model, self.n_assets * self.action_atoms)

    def forward(self, state_emb: torch.Tensor):
        qvals = self.out(state_emb).view(state_emb.shape[0],
                                         state_emb.shape[1], self.n_assets,
                                         self.action_atoms)
        return qvals  #(bs, nTau, n_assets, action_atoms)


class DuelingHeadIQN(nn.Module):
    def __init__(self,
                 d_model: int,
                 output_shape: tuple,
                 noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5):
        super().__init__()
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        self.value_net = Linear(d_model, self.n_assets)
        self.adv_net = Linear(d_model, self.n_assets * self.action_atoms)

    def forward(self, state_emb):
        value = self.value_net(state_emb)
        adv = self.adv_net(state_emb).view(state_emb.shape[0],
                                           state_emb.shape[1], self.n_assets,
                                           self.action_atoms)
        qvals = value[..., None] + adv - adv.mean(-1, keepdim=True)
        return qvals  #(bs, nTau, n_assets, action_atoms)


class TauEmbedLayer(nn.Module):
    """
    For use in Distributional DQN Approaches
    """
    def __init__(self,
                 d_embed: int,
                 d_model: int,
                 noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5,
                 device=None):
        super().__init__()
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        self.projection = Linear(d_embed, d_model)
        self.act = nn.GELU()
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.spectrum_embed = math.pi *\
            torch.arange(1, d_embed+1, dtype=torch.float32
                         ).view(1, 1, d_embed).to(device)

    def forward(self, tau):
        """
        positionally embeds tau, using cos basis and then creates a final embedding
        using self.projection
        """
        # if tau.device != self.spectrum_embed.device:
        #     self.spectrum_embed = self.spectrum_embed.to(tau.device)
        spectrum = tau[:, :, None] * self.spectrum_embed
        basis = torch.cos(spectrum)
        return self.act(self.projection(basis))


class ConvNetIQN(QNetworkBase):
    def __init__(self,
                 input_shape: tuple,
                 output_shape: tuple,
                 d_model: int = 512,
                 channels: list = [32, 32],
                 kernels: list = [5, 5],
                 strides: list = [1, 1],
                 dueling=True,
                 preserve_window_len: bool = False,
                 tau_embed_size=64,
                 nTau=32,
                 noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5,
                 **extra):
        """
        input_shape: (window_length, n_features)
        output_shape: (n_assets, action_atoms)
        """
        super().__init__()

        assert len(kernels) == len(strides) == len(channels)
        assert len(input_shape) == 2
        window_len = input_shape[0]
        assert window_len >= reduce(lambda x, y: x+y, kernels), \
            "window_length should be at least as long as sum of kernels"

        self.input_shape = input_shape
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        self.d_model = d_model
        self.act = nn.GELU()
        channels = [input_shape[1]] + channels
        conv_layers = []
        for i in range(len(kernels)):
            conv = nn.Conv1d(channels[i],
                             channels[i + 1],
                             kernels[i],
                             stride=strides[i])
            conv_layers.append(conv)
            if preserve_window_len:
                arb_input = (window_len, )
                # CAUSAL_DIM=0 assumes 0 is time dimension for input to calc_pad
                causal_pad = calc_pad_to_conserve(arb_input,
                                                  conv,
                                                  causal_dim=0)
                conv_layers.append(nn.ReplicationPad1d(causal_pad))
            conv_layers.append(self.act)
        self.conv_layers = nn.Sequential(*conv_layers)
        conv_out_shape = calc_conv_out_shape(window_len, self.conv_layers)
        self.price_project = nn.Linear(conv_out_shape[0] * channels[-1],
                                       d_model)
        self.port_project = nn.Linear(self.n_assets, d_model)
        self.tau_embed_layer = TauEmbedLayer(tau_embed_size, self.d_model,
                                             noisy_net=noisy_net,
                                             noisy_net_sigma=noisy_net_sigma)
        self.nTau = nTau
        self.noisy_net = noisy_net
        if dueling:
            self.output_head = DuelingHeadIQN(d_model, output_shape,
                                              noisy_net=noisy_net,
                                              noisy_net_sigma=noisy_net_sigma)
        else:
            self.output_head = NormalHeadIQN(d_model, output_shape,
                                             noisy_net=noisy_net,
                                             noisy_net_sigma=noisy_net_sigma)
        self.noisy_candidates = [self.tau_embed_layer, self.output_head]

    def get_state_emb(self, state: State):
        """
        Given a State object containing tensors as .price and .portfolio
        attributes, returns an embedding of shape (bs, d_model)
        """
        price = state.price.transpose(-1,
                                      -2)  # switch features and time dimension
        port = state.portfolio
        price_emb = self.conv_layers(price).view(price.shape[0], -1)
        price_emb = self.price_project(price_emb)
        port_emb = self.port_project(port)
        state_emb = price_emb * port_emb
        out = self.act(state_emb)
        return out

    def forward(self,
                state: State = None,
                state_emb: torch.Tensor = None,
                tau: torch.Tensor = None):
        """
        Returns qvals given either state or state_emb
        output_shape = (bs, n_assets, action_atoms)
        """
        assert state is not None or state_emb is not None
        if state_emb is None:
            state_emb = self.get_state_emb(state)  # (bs, d_model)
        if tau is None:
            tau = torch.rand(state.price.shape[0],
                             self.nTau,
                             dtype=torch.float32,
                             device=state_emb.device)
        tau_emb = self.tau_embed_layer(tau)
        state_emb = state_emb.unsqueeze(1) * tau_emb  # (bs, nTau, d_model)
        qvals = self.output_head(state_emb)
        return qvals  # (bs, nTau, n_assets, action_atoms)
