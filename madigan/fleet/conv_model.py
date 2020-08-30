from collections.abc import Iterable
import torch
import torch.nn as nn
from .model import Model

def calc_conv_out_shape(in_shape, layers):
    """
    Calculates output shape of input_shape going through a list of pytorch convolutional layers
    in_shape: (H, W)
    layers: list of convolution layers
    """
    shape = in_shape
    if isinstance(shape, Iterable):
        if len(shape) == 2:
            for layer in layers:
                if isinstance(layer, nn.Conv2d):
                    h_out = ((shape[0] + 2*layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1)-1) / layer.stride[0])+1
                    w_out = ((shape[1] + 2*layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1)-1) / layer.stride[1])+1
                    shape = (int(h_out), int(w_out))
        elif len(shape) == 1:
            raise NotImplementedError("1d conv not yet implemented")
            for layer in layers:
                if isinstance(layer, nn.Conv1d):
                    out = ((shape[0] + 2*layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1)-1) / layer.stride[0])+1
                    shape = (int(out), )
    elif isinstance(shape, int):
        return calc_conv_out_shape((shape, ), layers)
    else:
        raise ValueError("in_shape must be an iterable or int (for conv1d)")
    return shape


class PortEmbed(nn.Module):
    def __init__(self, n_assets, d_model):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.embed = nn.Linear(n_assets, d_model)
    def forward(self, raw_port):
        return self.embed(raw_port)



class OutputHead(nn.Module):
    def __init__(self, n_assets, d_model, action_atoms):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.action_atoms = action_atoms
        self.out = nn.Linear(2*d_model, n_assets*action_atoms)
    def forward(self, state_emb):
        return self.out(state_emb).view(state_emb.shape[0], self.n_assets, self.action_atoms)


class ConvModel(Model):
    def __init__(self, *, n_feats, min_tf, n_assets, action_atoms, d_model=256, channels=[32, 64, 64], kernels=[8, 4, 3],
                 strides=[3, 2, 1], act=nn.ReLU, **params):
        super().__init__()
        assert len(kernels) == len(strides) == len(channels)
        self.n_feats = n_feats
        self.action_atoms = action_atoms
        self.d_model = d_model
        self.act = act()
        channels = [n_feats] + channels
        conv_layers = []
        for i in range(len(kernels)):
            conv_layers.append(nn.Conv1d(channels[i], channels[i+1], kernels[i], stride=strides[i]))
            conv_layers.append(self.act)
        self.conv_layers = nn.Sequential(*conv_layers)
        conv_out_shape = calc_conv_out_shape((min_tf, n_assets), self.conv_layers)
        self.project = nn.Linear(conv_out_shape[0]*conv_out_shape[1], d_model)
        self.port_embed = nn.Linear(n_assets, d_model)
        self.out_head = OutputHead(n_assets, d_model, action_atoms)

    def forward(self, price, port, state_emb=None, price_emb=None, port_emb=None):
        if state_emb is None:
            state_emb = self.get_state_emb(price, port, price_emb=price_emb, port_emb=port_emb)
        return self.out_head(state_emb, port)

    def get_state_emb(self, price=None, port=None, price_emb=None, port_emb=None):
        assert price is not None or price_emb is not None, "Either price or price_emb must be passed"
        assert port is not None or port_emb is not None, "Either port or port_emb must be passed"

        price_emb = price_emb if price_emb is not None else self.get_price_emb(price)
        port_emb = port_emb if port_emb is not None else self.port_emb(price)

        return torch.cat([price_emb, port_emb], dim=-1)

    def get_price_emb(self, price):
        price_emb = price
        for layer in [self.c1, self.c2, self.c3]:
            price_emb = self.act(layer(price_emb))
        return self.act(self.fc(price_emb.view(price_emb.shape[0], -1)))


