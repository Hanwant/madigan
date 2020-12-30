from functools import reduce, partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import QNetworkBase
from .common import Conv1DEncoder, NoisyLinear
from .utils import xavier_initialization, orthogonal_initialization
from .utils import ACT_FN_DICT
from ...utils.data import StateRecurrent, SARSDR, StateRecurrent


class NormalHead(nn.Module):
    def __init__(self,
                 d_model,
                 output_shape,
                 noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5):
        super().__init__()
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        self.out = Linear(d_model, self.n_assets * self.action_atoms)

    def forward(self, state_emb):
        qvals = self.out(state_emb).view(state_emb.shape[0],
                                         state_emb.shape[1], self.n_assets,
                                         self.action_atoms)
        return qvals


class DuelingHead(nn.Module):
    def __init__(self,
                 d_model,
                 output_shape,
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
        bs = state_emb.shape[0]
        seq_len = state_emb.shape[1]
        value = self.value_net(state_emb)
        # adv = self.adv_net(state_emb).view(
        # bs, self.n_assets, self.action_atoms)
        # qvals = value[..., None] + adv - adv.mean(-1, keepdim=True)
        # return qvals
        adv = self.adv_net(state_emb)
        qvals = value + adv - adv.mean(-1, keepdim=True)
        return qvals.view(bs, seq_len, self.n_assets, self.action_atoms)


class LSTMStateEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, n_assets: int,
                 action_atoms: int, n_rewards: int, noisy_net: bool,
                 noisy_net_sigma: float, act_fn: str):
        super().__init__()
        self.n_assets = n_assets
        self.action_atoms = action_atoms
        self.num_layers = num_layers
        self.layers = nn.LSTM(d_model, d_model, num_layers, batch_first=True)
        self.noisy_net = noisy_net
        self.noisy_net_sigma = noisy_net_sigma
        self.act_fn = ACT_FN_DICT[act_fn]()

        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        input_size = n_assets + (action_atoms*n_assets) + \
            n_rewards + 1 # +1 for cash entry in portfolio vector
        self.project_in = Linear(input_size, d_model)

    def prep_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        input : Integer Vector (bs, seq_len, n_assets)
        output: One Hot Vector (bs, seq_len, n_assets*action_atoms)
        """
        one_hot = F.one_hot(action, self.action_atoms).to(action.device)
        return one_hot.flatten(-2, -1)

    def prep_reward(self, reward: torch.Tensor) -> torch.Tensor:
        return reward.unsqueeze(-1)  # add dim at end

    def forward(self, conv_emb: torch.Tensor,
                state: StateRecurrent) -> torch.Tensor:
        """
        All inputs are of shape (bs, seq_len, -1) -1 being feature size for
        that particular input

        IMPORTANT NOTE.
        Care must be taken to prevent lookahead bias, esp with regards to rewards.
        The actions and rewards provided from the replay buffer co-occur with each
        state but should only be passed as inputs for subsequent timesteps.
        So rewards and actions must must be sliced as [:-1]
        and price+port embeddings must be sliced as [1:]
        So that reward and action for timestep i
        are provided to state_emb for timestep i+1
        Portfolio can be provided to the concurring price embedding as the info
        is available to the agent at that timestep.
        """
        bs, rec_seq_len, _, _ = state.price.shape
        conv_emb = conv_emb.view(bs, rec_seq_len, -1)  # convert from 2d to 3d
        x = torch.cat([state.portfolio, state.action, state.reward], dim=-1)
        x = self.project_in(x)
        x = conv_emb * x
        x = self.act_fn(x)
        state_emb, (hn, cn) = self.layers(x, state.hidden)
        return state_emb, (hn.detach(), cn.detach())


class LSTMNet(QNetworkBase):
    """
    For use in DQN. Wraps ConvNetStateEncoder and adds a linear layer on top
    as an output head. May be a normal or dueling head.
    """
    def __init__(self,
                 input_shape: tuple,
                 output_shape: tuple,
                 n_recurrent_layers: int = 1,
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
        self.n_layers = n_recurrent_layers
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
            act_fn=act_fn)
        self.recurrent_encoder = LSTMStateEncoder(d_model, 1, self.n_assets,
                                                  self.action_atoms,
                                                  self.n_rewards,
                                                  self.noisy_net,
                                                  self.noisy_net_sigma, act_fn)
        self.layers = nn.LSTM(d_model,
                              d_model,
                              n_recurrent_layers,
                              batch_first=True)
        self.noisy_net = noisy_net
        self.noisy_net_sigma = noisy_net_sigma
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        input_size = self.n_assets + (self.action_atoms*self.n_assets) + \
            n_rewards + 1 # +1 for cash entry in portfolio vector
        self.project_in = Linear(input_size, d_model)
        if dueling:
            self.output_head = DuelingHead(d_model, output_shape, noisy_net,
                                           noisy_net_sigma)
        else:
            self.output_head = NormalHead(d_model, output_shape, noisy_net,
                                          noisy_net_sigma)

    def get_default_hidden(self, batch_size):
        return tuple(torch.zeros(self.n_layers, batch_size, self.d_model)
                     for _ in range(2))

    def get_state_emb(self, state: StateRecurrent) -> torch.Tensor:
        """
        Given a StateRecurrent object containing tensors for attributes
        price, portfolio, action, reward and hidden
        attributes, returns an embedding of shape (bs, d_model)
        Each element in sarsd should be of len seq_len
        """
        bs, rec_seq_len, seq_len, feats = state.price.shape
        price = state.price.transpose(-1, -2).reshape(bs * rec_seq_len, feats,
                                                      seq_len)
        conv_emb = self.conv_encoder(price)  # (bs, d_model)
        state_emb, (hn, cn) = self.recurrent_encoder(conv_emb, state)
        return state_emb, (hn.detach(), cn.detach())

    def forward(self,
                state: StateRecurrent = None,
                state_emb: torch.Tensor = None):
        """
        Returns qvals given either state or state_emb
        output_shape = (bs, n_assets, action_atoms)
        """
        assert state is not None or state_emb is not None
        state_emb, (hn, cn) = self.get_state_emb(state)  # (bs, d_model)
        qvals = self.output_head(state_emb)  # (bs, n_assets, action_atoms)
        return qvals, (hn, cn)
