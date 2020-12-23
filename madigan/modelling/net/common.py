import math
from functools import partial, reduce

import torch
import torch.nn as nn
from torch.nn.functional import linear as linear_func

from .utils import ACT_FN_DICT, calc_pad_to_conserve
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

class NormalHeadDQN(nn.Module):
    def __init__(self, d_model, output_shape, noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5):
        super().__init__()
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        self.out = Linear(d_model, self.n_assets*self.action_atoms)

    def forward(self, state_emb):
        qvals = self.out(state_emb).view(state_emb.shape[0],
                                         self.n_assets, self.action_atoms)
        return qvals

class DuelingHeadDQN(nn.Module):
    def __init__(self, d_model, output_shape, noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5):
        super().__init__()
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear
        # self.value_net = Linear(d_model, self.n_assets)
        self.value_net = Linear(d_model, 1)
        self.adv_net = Linear(d_model, self.n_assets*self.action_atoms)

    def forward(self, state_emb):
        bs = state_emb.shape[0]
        value = self.value_net(state_emb)
        # adv = self.adv_net(state_emb).view(
        # bs, self.n_assets, self.action_atoms)
        # qvals = value[..., None] + adv - adv.mean(-1, keepdim=True)
        # return qvals
        adv = self.adv_net(state_emb)
        qvals = value + adv - adv.mean(-1, keepdim=True)
        # import ipdb; ipdb.set_trace()
        return qvals.view(bs, self.n_assets, self.action_atoms)

class NoisyLinear(nn.Module):
    """
    Drop in replacement for normal linear layers
    Uses factorized gaussian noise - more efficient than independent noise
    Original in (Fortunato et al, 2017) https://arxiv.org/pdf/1706.10295.pdf
    Adaptation in (Hessel et al, 2017) https://arxiv.org/pdf/1710.02298.pdf
    """
    def __init__(self, in_feats, out_feats, sigma=0.5):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.sigma = sigma
        # learned params
        self.mu_w = nn.Parameter(torch.FloatTensor(out_feats, in_feats))
        self.sigma_w = nn.Parameter(torch.FloatTensor(out_feats, in_feats))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_feats))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_feats))
        # factorized gaussian params
        self.register_buffer('eps_p', torch.FloatTensor(in_feats))
        self.register_buffer('eps_q', torch.FloatTensor(out_feats))

        self.reset()
        self.sample()

    def reset(self):
        bound = 1 / math.sqrt(self.in_feats)
        self.mu_w.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_w.data.fill_(self.sigma / math.sqrt(self.in_feats))
        self.sigma_bias.data.fill_(self.sigma / math.sqrt(self.out_feats))

    def factorize(self, x):
        return x.normal_().sign().mul(x.abs().sqrt())

    def sample(self):
        self.eps_p.copy_(self.factorize(self.eps_p))
        self.eps_q.copy_(self.factorize(self.eps_q))

    def forward(self, x):
        if self.training:
            weight = self.mu_w + self.sigma_w * torch.ger(self.eps_q, self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_w
            bias = self.mu_bias
        return linear_func(x, weight, bias)


class Conv1DLayer(nn.Module):
    """
    Block unit for architectures using 1d Convs
    """
    def __init__(self, input_shape: tuple, channels_in: int, channels_out: int,
                 kernel: int, stride: int = None, dilation: int = None,
                 preserve_window_len: bool = False, act_fn: str = 'gelu',
                 causal_dim: int = 0):
        super().__init__()
        window_len = input_shape[0]
        self.conv = nn.Conv1d(channels_in, channels_out, kernel, stride=stride,
                              dilation=dilation)
        self.preserve_window_len = preserve_window_len
        self.pad = None
        if self.preserve_window_len:
            arb_input = (window_len, )
            causal_pad = calc_pad_to_conserve(arb_input, self.conv,
                                              causal_dim=causal_dim)
            self.pad = nn.ReplicationPad1d(causal_pad)
        self.act = ACT_FN_DICT[act_fn]()
        # self.norm = nn.BatchNorm1d(channels_out)
        self.pool = nn.MaxPool1d(kernel, stride=stride)
        # self.pool = lambda x: x
        self.norm = lambda x: x

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(self.act(x))
        if self.preserve_window_len:
            x = self.pad(x)
        x = self.norm(x)
        return x

class Conv1DEncoder(nn.Module):
    """
    Wraps and orchestrates a sequence of Conv1DLayers
    """
    def __init__(self, input_shape: tuple, channels: list, kernels: list,
                 strides: list = None, dilations: list = None,
                 preserve_window_len: bool = False, act_fn: str = 'gelu',
                 causal_dim: int = 0):
        super().__init__()
        if strides is None:
            strides = [1 for i in range(len(kernels))]
        assert len(kernels) == len(strides) == len(channels) == len(dilations)
        assert len(input_shape) == 2
        window_len = input_shape[0]
        input_feats = input_shape[1]
        channels = [input_feats] + channels
        layers = []
        for i, kernel in enumerate(kernels):
            layers.append(Conv1DLayer(input_shape, channels[i],
                                      channels[i+1],
                                      kernel, strides[i], dilations[i],
                                      preserve_window_len, act_fn,
                                      causal_dim))
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvNetStateEncoder(nn.Module):
    """
    Wraps Conv1DEncoder and integrates conditional information I.e portfolio
    Note that for noisy linear layers to be registered automatically, this
    must be wrapped in a class inheriting from QNetworkBase.

    Can be used in AC agents I.e DDPG - noisy_net will be false by default

    """
    def __init__(self,
                 input_shape: tuple,
                 n_assets: int,
                 d_model: int = 512,
                 channels: list = [32, 32],
                 kernels: list = [5, 5],
                 strides: list = [1, 1],
                 dilations: list = [1, 1],
                 preserve_window_len: bool = False,
                 act_fn: str = 'silu',
                 noisy_net: bool = False,
                 noisy_net_sigma: float = 0.5,
                 **extra):
        """
        input_shape: (window_length, n_features)
        """
        super().__init__()

        window_len = input_shape[0]
        assert window_len >= reduce(lambda x, y: x+y, kernels), \
            "window_length should be at least as long as sum of kernels"

        self.input_shape = input_shape
        self.window_len = window_len
        self.n_assets = n_assets
        self.d_model = d_model
        self.act = ACT_FN_DICT[act_fn]()
        self.conv_encoder = Conv1DEncoder(
            input_shape,
            channels,
            kernels,
            strides=strides,
            dilations=dilations,
            preserve_window_len=preserve_window_len,
            act_fn=act_fn,
            causal_dim=0)

        self.noisy_net = noisy_net
        self.noisy_net_sigma = noisy_net_sigma
        Linear = partial(NoisyLinear, sigma=noisy_net_sigma) \
            if noisy_net else nn.Linear

        # dummy_input = torch.randn(1, *input_shape[::-1])
        # conv_out_shape = self.conv_encoder(dummy_input).shape
        # conv_out_size = conv_out_shape[-1] * conv_out_shape[-2]
        pool_size = d_model // channels[-1]
        # print(conv_out_shape, conv_out_size, pool_size, d_model)
        self.price_pool = nn.AdaptiveMaxPool1d(pool_size)

        # normalized portfolio vector fed to model is number of assets + 1
        # +1 for cash (base currency) which is also included in the vector
        self.port_project = Linear(self.n_assets + 1, d_model)

    def forward(self, state: State):
        """
        Given a State object containing tensors as .price and .portfolio
        attributes, returns an embedding of shape (bs, d_model)
        """
        price = state.price.transpose(-1,
                                      -2)  # switch features and time dimension
        port = state.portfolio
        price_emb = self.conv_encoder(price)
        price_emb = self.price_pool(price_emb).view(price.shape[0], -1)
        # price_emb = self.price_project(price_emb)
        port_emb = self.port_project(port)
        state_emb = price_emb * port_emb
        out = self.act(state_emb)
        return out
