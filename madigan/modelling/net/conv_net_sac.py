from functools import reduce, partial

import torch
import torch.nn as nn

from .base import QNetworkBase
from .common import NoisyLinear, Conv1DEncoder, ConvNetStateEncoder
from .utils import ACT_FN_DICT, orthogonal_initialization
from .conv_net import ConvNetStateEncoder, ConvNet
from ...utils.data import State


ConvCriticQ = ConvNet

class TwinQNetwork(nn.Module):
    def __init__(self, *args, **kw):
        super().__init__()
        self.Q1 = ConvCriticQ(*args, **kw)
        self.Q2 = ConvCriticQ(*args, **kw)

    def forward(self, state: State):
        q1 = self.Q1(state)
        q2 = self.Q2(state)
        return q1, q2

class ConvPolicySACD(nn.Module):
    """
    Actor used in Discrete Soft Actor Critic.
    Implementatation follows https://github.com/ku2482/sac-discrete.pytorch
    """
    def __init__(self,
                 input_shape: tuple,
                 output_shape: tuple,
                 d_model: int = 512,
                 channels: list = [32, 32],
                 kernels: list = [5, 5],
                 strides: list = [1, 1],
                 preserve_window_len: bool = False,
                 act_fn: str = 'gelu',
                 **extra):
        super().__init__()
        window_len = input_shape[0]
        assert window_len >= reduce(lambda x, y: x+y, kernels), \
            "window_length should be at least as long as sum of kernels"

        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        self.d_model = d_model
        self.act = ACT_FN_DICT[act_fn]()
        self.convnet_state_encoder = ConvNetStateEncoder(
            input_shape,
            self.n_assets,
            d_model,
            channels,
            kernels,
            strides,
            preserve_window_len=preserve_window_len,
            act_fn=act_fn,
            noisy_net=False,
            noisy_net_sigma=0.,
            **extra)

        # conv_out_shape = calc_conv_out_shape(window_len, self.conv_encoder)
        self.output_head = nn.Linear(d_model, self.n_assets*self.action_atoms)
        self.apply(orthogonal_initialization)

    def get_state_emb(self, state: State):
        """
        Given a State object containing tensors as .price and .portfolio
        attributes, returns an embedding of shape (bs, d_model)
        """
        return self.convnet_state_encoder(state)

    def get_action(self, state: State = None, state_emb: torch.Tensor = None):
        """
        Returns qvals given either state or state_emb
        output_shape = (bs, n_assets, action_atoms)
        """
        assert (state is not None) or (state_emb is not None)
        if state_emb is None:
            state_emb = self.get_state_emb(state)  # (bs, d_model)
        action_logits = self.output_head(state_emb).view(state_emb.shape[0],
                                                         self.n_assets,
                                                         self.action_atoms)
        greedy_actions = action_logits.argmax(dim=-1, keepdim=True)
        return greedy_actions

    def sample(self, state: State = None, state_emb: torch.Tensor = None):
        assert (state is not None) or (state_emb is not None), "pass state or state_emb"
        if state_emb is None:
            state_emb = self.get_state_emb(state)  # (bs, d_model)
        action_logits = self.output_head(state_emb).view(
            state_emb.shape[0], self.n_assets, self.action_atoms)
        action_p = nn.functional.softmax(action_logits, dim=2)
        action_dist = torch.distributions.Categorical(action_p)
        actions = action_dist.sample()

        # avoids numerical instability
        z = (action_p == .0).float() * 1e-8
        log_action_p = (action_p + z).log()
        return actions, action_p, log_action_p

    def forward(self, state: State = None, state_emb: torch.Tensor = None):
        return self.get_action(state=state, state_emb=state_emb)

