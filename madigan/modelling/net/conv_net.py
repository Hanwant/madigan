from functools import reduce, partial
from collections.abc import Iterable
import math

import torch
import torch.nn as nn

from .base import QNetworkBase
from .common import NoisyLinear, NormalHeadDQN, DuelingHeadDQN
from .utils import Conv1DEncoder, calc_conv_out_shape
from .utils import calc_pad_to_conserve, ACT_FN_DICT
from .utils import xavier_initialization, orthogonal_initialization
from ...utils.data import State


class ConvNet(QNetworkBase):
    """
    For use in DQN
    """
    def __init__(self,
                 input_shape: tuple,
                 output_shape: tuple,
                 d_model=512,
                 channels=[32, 32],
                 kernels=[5, 5],
                 strides=[1, 1],
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

        window_len = input_shape[0]
        assert window_len >= reduce(lambda x, y: x+y, kernels), \
            "window_length should be at least as long as sum of kernels"

        self.input_shape = input_shape
        self.window_len = window_len
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        self.d_model = d_model
        self.act = ACT_FN_DICT[act_fn]()
        # self.conv_layers = make_conv1d_layers(
        #     input_shape,
        #     kernels,
        #     channels,
        #     strides=strides,
        #     act=nn.GELU,
        #     preserve_window_len=preserve_window_len,
        #     causal_dim=0)
        self.conv_encoder = Conv1DEncoder(
            input_shape,
            kernels,
            channels,
            strides=strides,
            preserve_window_len=preserve_window_len,
            act_fn=act_fn,
            causal_dim=0)

        self.noisy_net = noisy_net
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear

        dummy_input = torch.randn(1, *input_shape[::-1])
        conv_out_shape = self.conv_encoder(dummy_input).shape
        self.price_project = Linear(conv_out_shape[-1] * conv_out_shape[-2],
                                    d_model)
        # normalized portfolio vector fed to model is number of assets + 1
        # +1 for cash (base currency) which is also included in the vector
        self.port_project = Linear(self.n_assets + 1, d_model)
        if dueling:
            self.output_head = DuelingHeadDQN(d_model,
                                              output_shape,
                                              noisy_net=noisy_net)
        else:
            self.output_head = NormalHeadDQN(d_model,
                                             output_shape,
                                             noisy_net=noisy_net)
        # self.register_noisy_layers()

    def get_state_emb(self, state: State):
        """
        Given a State object containing tensors as .price and .portfolio
        attributes, returns an embedding of shape (bs, d_model)
        """
        price = state.price.transpose(-1,
                                      -2)  # switch features and time dimension
        port = state.portfolio
        price_emb = self.conv_encoder(price).view(price.shape[0], -1)
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
            state_emb = self.get_state_emb(state)  # (bs, d_model)
        qvals = self.output_head(state_emb)  # (bs, n_assets, action_atoms)
        return qvals

class ConvNetCurl(ConvNet):
    def __init__(self,
                 input_shape: tuple,
                 output_shape: tuple,
                 d_model=512,
                 channels=[32, 32],
                 kernels=[5, 5],
                 strides=[1, 1],
                 dueling=True,
                 preserve_window_len: bool = False,
                 noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5,
                 random_crop_ratio=.84**2,
                 **extra):
        super().__init__(input_shape, output_shape, d_model, channels, kernels,
                         strides, dueling, preserve_window_len, noisy_net,
                         noisy_net_sigma)
        self.random_crop_ratio = random_crop_ratio
        self.random_crop_length = math.floor(self.window_len *
                                             self.random_crop_ratio)
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        dummy_input = torch.randn(1, *input_shape)
        conv_out_len = self.conv_layers(dummy_input).shape[-1]
        self.query_project = Linear(int(conv_out_len * self.random_crop_ratio *
                                        channels[-1]), d_model)
    def data_transform(self, x: torch.Tensor):
        """
        Performs a random crop taking the SAME RANDOM SLICE across batches.
        Although independent slices would be best, this is more efficient
        and should (hopefullly) not introduce much bias.
        RECONSIDER.
        args:
          x: torch.Tensor  -shape (batch, channels, time) where the time
                            dimension is of length self.window_length
        returns:
          torch.Tensor of shape (batch, channels, time) where size of time is
          self.random_crop_length

        """
        idx = torch.randint(0, self.window_length-self.random_crop_length, (1,))
        x = x[..., idx: idx+self.random_crop_length]
        return x

    def encode_query(self, state: State):
        """
        Takes same input as get_state_embedding but only considers
        Differs from get_state_embedding as this crops the input before encoding
        hence it
        """
        price = state.price.transpose(-1,
                                      -2)  # switch features and time dimension
        # port = state.portfolio
        price_emb = self.conv_layers(price).view(price.shape[0], -1)
        price_emb = self.query_project(price_emb)
        # port_emb = self.port_project(port)
        # state_emb = price_emb * port_emb
        out = self.act(price_emb)
        return out



class ConvCriticQ(nn.Module):
    """
    For use as critic in Actor Critic methods, takes both state and action as input
    """
    def __init__(self,
                 input_shape: tuple,
                 n_assets: int,
                 n_actions: int = 1,
                 d_model=512,
                 channels=[32, 32],
                 kernels=[5, 5],
                 strides=[1, 1],
                 preserve_window_len: bool = False,
                 act_fn: str = 'silu',
                 noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5,
                 **extra):
        super().__init__()
        window_len = input_shape[0]
        assert window_len >= reduce(lambda x, y: x+y, kernels), \
            "window_length should be at least as long as sum of kernels"

        self.input_shape = input_shape
        self.n_assets = n_assets
        self.n_actions = n_actions
        self.d_model = d_model
        self.act = ACT_FN_DICT[act_fn]()
        # self.conv_layers = make_conv1d_layers(
        #     input_shape,
        #     kernels,
        #     channels,
        #     strides=strides,
        #     act=nn.GELU,
        #     preserve_window_len=preserve_window_len,
        #     causal_dim=0)
        self.conv_encoder = Conv1DEncoder(
            input_shape,
            kernels,
            channels,
            strides=strides,
            preserve_window_len=preserve_window_len,
            act_fn=act_fn,
            causal_dim=0)
        # conv_out_shape = calc_conv_out_shape(window_len, self.conv_encoder)
        # self.price_project = nn.Linear(conv_out_shape[0]*channels[-1], d_model)
        # self.port_project = nn.Linear(self.n_assets, d_model)
        # self.action_project = nn.Linear(self.n_assets, d_model)
        self.noisy_net = noisy_net
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear

        dummy_input = torch.randn(1, *input_shape[::-1])
        conv_out_shape = self.conv_encoder(dummy_input).shape
        project_in = conv_out_shape[-1] * conv_out_shape[-2] + 2 * self.n_assets
        self.projection = nn.Linear(project_in, d_model)
        self.output_head = Linear(d_model, 1)
        self.register_noisy_layers()
        self.apply(orthogonal_initialization)


    def get_state_emb(self, state: State, action: torch.Tensor):
        """
        Given a State object containing tensors as .price and .portfolio
        attributes, returns an embedding of shape (bs, d_model)
        """
        price = state.price.transpose(-1,
                                      -2)  # switch features and time dimension
        port = state.portfolio
        price_emb = self.conv_encoder(price).view(price.shape[0], -1)
        # price_emb = self.price_project(price_emb)
        # port_emb = self.port_project(port)
        # action_emb = self.action_project(action)
        state_emb = torch.cat([price_emb, port, action], dim=-1)
        state_emb = self.projection(state_emb)
        out = self.act(state_emb)
        return out

    def forward(self,
                state: State = None,
                action: torch.Tensor = None,
                state_emb: torch.Tensor = None):
        """
        Returns qvals given either state or state_emb
        output_shape = (bs, n_assets, action_atoms)
        """
        assert (None not in (state, action)) or state_emb is not None
        if state_emb is None:
            state_emb = self.get_state_emb(state, action)  # (bs, d_model)
        qvals = self.output_head(state_emb)  # (bs, n_assets, action_atoms)
        return qvals


class ConvPolicyDeterministic(nn.Module):
    """
    For use as actor in Actor Critic methods, takes both state and action as input
    """
    def __init__(self,
                 input_shape: tuple,
                 n_assets: int,
                 n_actions: int = 1,
                 d_model=512,
                 channels=[32, 32],
                 kernels=[5, 5],
                 strides=[1, 1],
                 dueling=True,
                 preserve_window_len: bool = False,
                 act_fn: str = 'gelu',
                 **extra):
        super().__init__()
        window_len = input_shape[0]
        assert window_len >= reduce(lambda x, y: x+y, kernels), \
            "window_length should be at least as long as sum of kernels"

        self.input_shape = input_shape
        self.n_assets = n_assets
        self.n_actions = n_actions
        assert self.n_actions == 1
        self.d_model = d_model
        self.act = ACT_FN_DICT[act_fn]()
        # self.conv_layers = make_conv1d_layers(
        #     input_shape,
        #     kernels,
        #     channels,
        #     strides=strides,
        #     act=self.act_fn,
        #     preserve_window_len=preserve_window_len,
        #     causal_dim=0)
        self.conv_encoder = Conv1DEncoder(
            input_shape,
            kernels,
            channels,
            strides=strides,
            preserve_window_len=preserve_window_len,
            act_fn=act_fn,
            causal_dim=0)

        # conv_out_shape = calc_conv_out_shape(window_len, self.conv_encoder)
        dummy_input = torch.randn(1, *input_shape[::-1])
        conv_out_shape = self.conv_encoder(dummy_input).shape
        project_in = conv_out_shape[0] * channels[-1] + self.n_assets
        self.projection = nn.Linear(project_in, d_model)
        self.output_head = nn.Linear(d_model, self.n_assets * self.n_actions)
        # self.apply(self.initialize_weights)
        # self.apply(xavier_initialization, linear_range=(-3e-4, 3e-4))
        self.apply(orthogonal_initialization)

    def get_state_emb(self, state: State):
        """
        Given a State object containing tensors as .price and .portfolio
        attributes, returns an embedding of shape (bs, d_model)
        """
        price = state.price.transpose(-1,
                                      -2)  # switch features and time dimension
        port = state.portfolio
        price_emb = self.conv_encoder(price).view(price.shape[0], -1)
        # price_emb = self.price_project(price_emb)
        # port_emb = self.port_project(port)
        state_emb = torch.cat([price_emb, port], dim=-1)
        # import ipdb; ipdb.set_trace()
        state_emb = self.projection(state_emb)
        out = self.act(state_emb)
        return out

    def forward(self, state: State = None, state_emb: torch.Tensor = None):
        """
        Returns qvals given either state or state_emb
        output_shape = (bs, n_assets, action_atoms)
        """
        assert (None not in (state, )) or state_emb is not None
        if state_emb is None:
            state_emb = self.get_state_emb(state)  # (bs, d_model)
        actions = self.output_head(state_emb).view(state_emb.shape[0],
                                                   self.n_assets,
                                                   self.n_actions)
        return torch.tanh(actions)
        # return torch.sigmoid(actions)
