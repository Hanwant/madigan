from functools import reduce, partial
from collections.abc import Iterable
import math
import copy

import numpy as np
import torch
import torch.nn as nn

from .base import QNetworkBase
from .common import NoisyLinear, NormalHeadDQN, DuelingHeadDQN, Conv1DEncoder
from .common import ConvNetStateEncoder
from .utils import calc_conv_out_shape
from .utils import calc_pad_to_conserve, ACT_FN_DICT
from .utils import xavier_initialization, orthogonal_initialization
from ...utils.data import State




class ConvNet(QNetworkBase):
    """
    For use in DQN. Wraps ConvNetStateEncoder and adds a linear layer on top
    as an output head. May be a normal or dueling head.
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

        assert len(input_shape) == 2, \
            "input_shape should be (window_len, n_feats)"
        window_len = input_shape[0]
        assert window_len >= reduce(lambda x, y: x+y, kernels), \
            "window_length should be at least as long as sum of kernels"

        self.input_shape = input_shape
        self.window_len = window_len
        self.n_assets = output_shape[0]
        self.action_atoms = output_shape[1]
        self.d_model = d_model
        self.act = ACT_FN_DICT[act_fn]()
        self.noisy_net = noisy_net
        self.noisy_net_sigma = noisy_net_sigma
        self.convnet_state_encoder = ConvNetStateEncoder(
            input_shape,
            self.n_assets,
            d_model,
            channels,
            kernels,
            strides,
            preserve_window_len=preserve_window_len,
            act_fn=act_fn,
            noisy_net=noisy_net,
            noisy_net_sigma=noisy_net_sigma,
            **extra)
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
        return self.convnet_state_encoder(state)

    def forward(self, state: State = None, state_emb: torch.Tensor = None):
        """
        Returns qvals given either state or state_emb
        output_shape = (bs, n_assets, action_atoms)
        """
        assert state is not None or state_emb is not None
        if state_emb is None:
            state_emb = self.convnet_state_encoder(state)  # (bs, d_model)
        qvals = self.output_head(state_emb)  # (bs, n_assets*action_atoms)
        return qvals


ConvNetDQN = ConvNet

class ConvNetAE(ConvNet):
    """
    Provides an interface allowing for training of an autoencoder objective.
    Can be used in an unsupervised setting where the AE objective is used
    as an auxiliary loss.

    Inherits all other functionaility (for normal rl objective) from ConvNet
    """
    def __init__(self,
                 input_shape: tuple,
                 output_shape: tuple,
                 compression_factor: int = 4,
                 **kw):
        super().__init__(input_shape, output_shape, **kw)
        self.compression_factor = compression_factor
        self.encoded_size = self.d_model // compression_factor
        self.compressor = nn.Linear(self.d_model, self.encoded_size)
        self.reconstruct_price = nn.Linear(self.encoded_size,
                                           input_shape[0] * input_shape[1])
        # portfolio.len = self.n_assets + 1 - +1 for cash
        self.reconstruct_port = nn.Linear(self.encoded_size, self.n_assets + 1)

    def reconstruct(self, state: State = None, state_emb: torch.Tensor = None):
        """
        Compresses an embedding of the state by self.compression_factor
        and returns a reconstruction from the compression.
        """
        assert state is not None or state_emb is not None
        if state_emb is None:
            state_emb = self.convnet_state_encoder(state)  # (bs, d_model)
        latent = self.compressor(state_emb)
        price_recon = self.reconstruct_price(latent).view(latent.shape[0],
                                                          *self.input_shape)
        port_recon = self.reconstruct_port(latent)
        return price_recon, port_recon

    def reconstruction_loss(self, state: State = None):
        price_recon, port_recon = self.reconstruct(state)
        price_loss = nn.functional.mse_loss(price_recon, state.price)
        port_loss = nn.functional.mse_loss(port_recon, state.portfolio)
        return price_loss + port_loss


class ConvNetCurl(ConvNet):
    """
    Inherits main RL functionality from ConvNet and augments with
    contrastive unsupervised representation learning (CURL).

    Introduces self.key_encoder and uses self.convnet_state_encoder as
    self.query_encoder. The difference being that keys and queries are
    encoded using a data augmentation (random crop) whereas get_state_emb
    for the rl objective still uses the full timeseries, while sharing
    weights with the query_encoder.

    The key encoder does not receive gradients and is updated by
    an exponentially weighted moving average of the query net much like the
    target is of the online network in DQN.

    """
    def __init__(self,
                 *args,
                 random_crop_ratio: float = .84**2,
                 curl_momentum_update: float = 0.001,
                 curl_latent_size: int = 64,
                 curl_lr: float = 1e-3,
                 **kw):
        super().__init__(*args, **kw)
        self.random_crop_ratio = random_crop_ratio
        self.random_crop_length = math.floor(self.window_len *
                                             self.random_crop_ratio)
        # safety check - gets 0.001 instead of 0.99
        self.momentum_update = min(curl_momentum_update,
                                   1 - curl_momentum_update)
        self.latent_project = nn.Linear(self.d_model, curl_latent_size)
        self.W = nn.Parameter(torch.randn(curl_latent_size, curl_latent_size),
                              requires_grad=True)

        self.query_encoder = self.convnet_state_encoder
        # add a projection from d_model to latent_size
        self.query_encoder.latent_project = self.latent_project

        self.curl_opt = torch.optim.Adam(self.parameters(), lr=curl_lr)

        # define after self.curl_opt so optimizer doesn't include key params
        self.key_encoder = copy.deepcopy(self.query_encoder)

        for module in self.key_encoder.modules():
            module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

        self.curl_loss_fn = nn.CrossEntropyLoss()

    @torch.no_grad()
    def update_key_encoder(self):
        """
        performs ema (exponential moving average) update from query encoder
        """
        for q_param, k_param in zip(self.query_encoder.parameters(),
                                    self.key_encoder.parameters()):
            k_param.data.copy_(self.momentum_update * q_param +
                               (1 - self.momentum_update) * k_param)

    def data_transform(self, x: torch.Tensor):
        """
        Performs a random crop independently for each batch
        args:
          x: torch.Tensor  - shape (batch, channels, time) where the time
                            dimension is of length self.window_length
        returns:
          torch.Tensor of shape (batch, channels, time) where size of time is
          self.random_crop_length

        """
        assert len(x.shape) == 3
        bs, window_len, nfeats = x.shape
        crop_len = int(window_len * self.random_crop_ratio)
        idx = np.random.randint(0, window_len - crop_len, bs)
        idx = np.r_[[np.r_[i: i + crop_len] for i in idx]]  # (bs, crop_len)
        idx = torch.LongTensor((idx + crop_len * np.arange(bs)[:, None]
                                ).flatten()).to(x.device)
        x = torch.index_select(x.view(-1, x.shape[-1]), 0, idx)
        return x.view(bs, crop_len, nfeats)

    def encode(self, state: State):
        """
        Takes same input as get_state_embedding but only considers
        Differs from get_state_embedding as this crops the input before encoding
        hence it
        """

    def get_state_emb(self, state: State):
        """
        Given a State object containing tensors as .price and .portfolio
        attributes, returns an embedding of shape (bs, d_model)
        Difference from normal: adds adaptive pool to share paramters
        when cropping vs taking full input
        """
        return self.convnet_state_encoder(state)

    def get_query_emb(self, state):
        """
        Same as get_state_emb except that it performs data augmentation.
        In the original curl, this is used instead of the full uncropped image
        to benefit from data augmentation. As recent information may be critical
        in timeseries, get_state_emb is used for the rl objective while
        this is used for the contrastive unsupervised representation objective.

        """
        # Get query embeddings first
        price = self.data_transform(state.price)
        # self.query_encoder == self.convnet_state_encoder
        query_state_emb = self.query_encoder(
            State(price, state.portfolio, state.timestamp))
        query_latent_emb = self.query_encoder.latent_project(query_state_emb)
        return query_latent_emb

    @torch.no_grad()
    def get_key_emb(self, state):
        """
        Same as get_query_emb expect uses self.key_encoder

        """
        # Get query embeddings first
        price = self.data_transform(state.price)
        key_state_emb = self.key_encoder(
            State(price, state.portfolio, state.timestamp))
        key_latent_emb = self.key_encoder.latent_project(key_state_emb)
        return key_latent_emb

    def train_contrastive_objective(self, state: State):
        """
        See section 4.7 of curl paper
        """
        bs = state.price.shape[0]
        queries = self.get_query_emb(state)
        keys = self.get_key_emb(state)
        proj_k = torch.matmul(self.W, keys.T)
        logits = torch.matmul(queries, proj_k)
        assert logits.shape == (bs, bs)

        logits = logits - logits.max(axis=1)[0]
        labels = torch.arange(bs).to(logits.device)
        loss = self.curl_loss_fn(logits, labels)
        loss.backward()
        self.curl_opt.step()
        self.update_key_encoder()
        return loss.mean().detach().item()



