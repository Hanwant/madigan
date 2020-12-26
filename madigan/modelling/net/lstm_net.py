from functools import reduce, partial

import torch
import torch.nn as nn

from .base import QNetworkBase
from .common import Conv1DEncoder, DuelingHead, NormalHead, NoisyLinear
from .utils import xavier_initialization, orthogonal_initialization
from .utils import ACT_FN_DICT
from ...utils.data import State, SARSD


class LSTMStateEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, n_assets: int,
                 n_actions: int, n_rewards: int, noisy_net: bool,
                 noisy_net_sigma: float):
        self.d_model = d_model
        self.num_layers = num_layers
        self.layers = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        self.noisy_net = noisy_net
        self.noisy_net_sigma = noisy_net_sigma
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        input_size = n_assets + 1 + n_actions + n_rewards  # +1 for cash entry in port
        self.project_in = Linear(input_size, d_model)

    def forward(self, conv_emb: torch.Tensor, portfolio: torch.Tensor,
                action: torch.Tensor, reward: torch.Tensor,
                hidden: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        x = torch.cat([portfolio, action, reward], dim=-1)
        x = self.project_in(x)
        x = conv_emb * x
        x = self.act_fn(x)
        state_emb, (hn, cn) = self.layers(x, (hidden, cell))
        return state_emb, (hn, cn)


class LSTMNet(QNetworkBase):
    """
    For use in DQN. Wraps ConvNetStateEncoder and adds a linear layer on top
    as an output head. May be a normal or dueling head.
    """
    def __init__(self,
                 input_shape: tuple,
                 output_shape: tuple,
                 n_rewards: int = 1,
                 d_model=512,
                 channels=[32, 32],
                 kernels=[5, 5],
                 strides=[1, 1],
                 dilations=[1, 1],
                 dueling=True,
                 preserve_window_len: bool = False,
                 act_fn: str = 'silu',
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
        self.window_len = window_len
        self.n_rewards = n_rewards
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        self.d_model = d_model
        self.act = ACT_FN_DICT[act_fn]()
        self.noisy_net = noisy_net
        self.noisy_net_sigma = noisy_net_sigma
        self.conv_encoder = Conv1DEncoder(
            input_shape,
            d_model,
            channels,
            kernels,
            strides,
            dilations,
            preserve_window_len=preserve_window_len,
            act_fn=act_fn,
            noisy_net=noisy_net,
            noisy_net_sigma=noisy_net_sigma,
            **extra)
        self.recurrent_encoder = LSTMStateEncoder(d_model, 1, self.n_assets,
                                                  self.action_atoms,
                                                  self.n_rewards,
                                                  self.noisy_net,
                                                  self.noisy_net_sigma)
        if dueling:
            self.output_head = DuelingHead(d_model, output_shape)
        else:
            self.output_head = NormalHead(d_model, output_shape)

    def get_state_emb(self, sarsd: SARSD) -> torch.Tensor:
        """
        Given a State object containing tensors as .price and .portfolio
        attributes, returns an embedding of shape (bs, d_model)
        Each element in sarsd should be of len seq_len
        """
        conv_emb = self.conv_encoder(sarsd.state)  # (bs, seq_len, d_model)
        state_emb, (h0, cell) = self.recurrent_encoder(conv_emb,
                                                       sarsd.state.portfolio,
                                                       sarsd.reward,
                                                       sarsd.action)
        return state_emb, (hn, cn)

    def forward(self, state: State = None, state_emb: torch.Tensor = None):
        """
        Returns qvals given either state or state_emb
        output_shape = (bs, n_assets, action_atoms)
        """
        assert state is not None or state_emb is not None
        if state_emb is None:
            state_emb, (hn, cn) = self.get_state_emb(state)  # (bs, d_model)
        qvals = self.output_head(state_emb)  # (bs, n_assets, action_atoms)
        return qvals, (hn, cn)
