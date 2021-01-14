import math
from functools import reduce, partial

import torch
import torch.nn as nn

from .common import NoisyLinear, Conv1DEncoder, ConvNetStateEncoder
from .base import QNetworkBase
from .utils import ACT_FN_DICT
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
        return qvals  # (bs, nTau, n_assets, action_atoms)


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
        # self.value_net = Linear(d_model, self.n_assets)
        self.value_net = Linear(d_model, 1)
        self.adv_net = Linear(d_model, self.n_assets * self.action_atoms)

    def forward(self, state_emb):
        value = self.value_net(state_emb)
        # adv = self.adv_net(state_emb).view(state_emb.shape[0],
        #                                    state_emb.shape[1], self.n_assets,
        #                                    self.action_atoms)
        # qvals = value[..., None] + adv - adv.mean(-1, keepdim=True)
        # return qvals  # (bs, nTau, n_assets, action_atoms)
        adv = self.adv_net(state_emb)
        qvals = value + adv - adv.mean(-1, keepdim=True)
        return qvals.view(state_emb.shape[0], state_emb.shape[1],
                          self.n_assets, self.action_atoms)


class TauEmbedLayer(nn.Module):
    """
    For use in Distributional DQN Approaches
    """
    def __init__(self,
                 d_embed: int,
                 d_model: int,
                 act_fn: str = 'gelu',
                 noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5,
                 device=None):
        super().__init__()
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        self.projection = Linear(d_embed, d_model)
        self.act = ACT_FN_DICT[act_fn]()
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
                 account_info_len: int,
                 d_model: int = 512,
                 channels: list = [32, 32],
                 kernels: list = [5, 5],
                 strides: list = [1, 1],
                 dueling=True,
                 preserve_window_len: bool = False,
                 act_fn: str = 'silu',
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

        assert len(input_shape) == 2, \
            "input_shape should be (window_len, n_feats)"
        window_len = input_shape[0]
        assert window_len >= reduce(lambda x, y: x+y, kernels), \
            "window_length should be at least as long as sum of kernels"

        self.input_shape = input_shape
        self.account_info_len = account_info_len
        self.action_atoms = output_shape[1]
        self.d_model = d_model
        self.act = ACT_FN_DICT[act_fn]()
        self.convnet_state_encoder = ConvNetStateEncoder(
            input_shape,
            self.account_info_len,
            d_model,
            channels,
            kernels,
            strides,
            preserve_window_len=preserve_window_len,
            act_fn=act_fn,
            noisy_net=noisy_net,
            noisy_net_sigma=noisy_net_sigma,
            **extra)

        self.noisy_net = noisy_net
        self.tau_embed_layer = TauEmbedLayer(tau_embed_size, self.d_model,
                                             noisy_net=noisy_net,
                                             noisy_net_sigma=noisy_net_sigma)
        self.nTau = nTau
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
        return self.convnet_state_encoder(state)

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
