from functools import reduce
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
    padding_classes2d = (nn.ConstantPad2d, nn.ReflectionPad2d, nn.ReplicationPad2d)
    padding_classes1d = (nn.ConstantPad1d, nn.ReflectionPad1d, nn.ReplicationPad1d)
    if isinstance(shape, Iterable):
        if len(shape) == 2:
            for layer in layers:
                if isinstance(layer, nn.Conv2d):
                    h_out = ((shape[0] + 2*layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1)-1) / layer.stride[0])+1
                    w_out = ((shape[1] + 2*layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1)-1) / layer.stride[1])+1
                    shape = (int(h_out), int(w_out))
                elif isinstance(layer, padding_classes2d):
                    h_out = shape[0] + layer.padding[0] + layer.padding[1]
                    w_out = shape[1] + layer.padding[2] + layer.padding[3]
                    shape = (int(h_out), int(w_out))
        elif len(shape) == 1:
            for layer in layers:
                if isinstance(layer, nn.Conv1d):
                    out = ((shape[0] + 2*layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1)-1) / layer.stride[0])+1
                    shape = (int(out),)
                elif isinstance(layer,  padding_classes1d):
                    out = shape[0] + layer.padding[0] + layer.padding[1]
                    shape = (int(out),)
    elif isinstance(shape, int):
        return calc_conv_out_shape((shape, ), layers)
    else:
        raise ValueError("in_shape must be an iterable or int (for conv1d)")
    return shape

def calc_pad_to_conserve(in_shape, layer, causal_dim=0):
    """
    Outputs a 4 item tuple for a 2d conv for (H_up, H_down, W_left, W_right)
    if causal_dim is specified, the padding is assymetric for that dim of the shape
    The padding is applied to the first element of the pair for that dimenion
    I.e
    in_shape = (64, 64), layer.kernel = (3, 3)
    with asymmetric causal padding (dim 0):
    to_pad = (2, 0, 1, 1)
    """
    out_shape = calc_conv_out_shape(in_shape, [layer, ])
    pad = []
    if isinstance(in_shape, Iterable):
        if len(in_shape) == 2:
            for i in range(len(in_shape)):
                diff = (in_shape[i] - out_shape[i]) / 2
                if causal_dim == i:
                    pad += [int(diff*2), 0]
                else:
                    pad += [int(diff), int(diff)]
            post_pad_input_shape = (pad[0]+pad[1]+in_shape[0], pad[2]+pad[3]+in_shape[1])
            post_pad_output_shape = calc_conv_out_shape(post_pad_input_shape, [layer, ])
        elif len(in_shape) == 1:
            diff = (in_shape[0] - out_shape[0]) / 2
            if causal_dim == 0:
                pad += [int(diff*2), 0]
            else:
                pad += [int(diff), int(diff)]
            post_pad_input_shape = (pad[0]+pad[1]+in_shape[0],)
        post_pad_output_shape = calc_conv_out_shape(post_pad_input_shape, [layer, ])
    elif isinstance(in_shape, int):
        return calc_pad_to_conserve((in_shape, ), layer, causal_dim=causal_dim)
    else:
        raise ValueError("in_shape must be an iterable or int")
    assert tuple(in_shape) == tuple(post_pad_output_shape)
    return tuple(pad)

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
        self.action_atoms = action_atoms
        self.out = nn.Sequential(nn.Linear(d_model, d_model),
                                 nn.ReLU(),
                                 nn.Linear(d_model, n_assets*action_atoms))
    def forward(self, state_emb):
        return self.out(state_emb).view(state_emb.shape[0], self.n_assets, self.action_atoms)

class ConvModel(Model):
    def __init__(self, *, min_tf, n_assets, action_atoms, d_model=256, channels=[32, 64, 64], kernels=[3, 3, 3],
                 strides=[1, 1, 1], act=nn.ReLU, **params):
        super().__init__()
        assert len(kernels) == len(strides) == len(channels)
        assert min_tf >= reduce(lambda x, y: x+y, kernels), "min_tf should be at least as long as sum of kernels"
        self.action_atoms = action_atoms
        self.d_model = d_model
        self.act = act()
        channels = [n_assets] + channels
        conv_layers = []
        for i in range(len(kernels)):
            conv = nn.Conv1d(channels[i], channels[i+1], kernels[i], stride=strides[i])
            conv_layers.append(conv)
            arb_input = (min_tf)
            causal_pad = calc_pad_to_conserve(arb_input, conv, causal_dim=0) # CAUSAL_DIM=0 assumes 0 is time dimension
            conv_layers.append(nn.ReplicationPad1d(causal_pad))
            conv_layers.append(self.act)
        self.conv_layers = nn.Sequential(*conv_layers)
        conv_out_shape = calc_conv_out_shape(min_tf, self.conv_layers)
        self.project = nn.Linear(conv_out_shape[0]*channels[-1], d_model)
        self.port_embed = nn.Linear(n_assets, d_model)
        self.out_head = OutputHead(n_assets, 2*d_model, action_atoms)

    def forward(self, state, state_emb=None, price_emb=None, port_emb=None):
        price, port = state.price.transpose(1, -1), state.portfolio
        if state_emb is None:
            state_emb = self.get_state_emb(price, port, price_emb=price_emb, port_emb=port_emb)
        return self.out_head(state_emb)

    def get_state_emb(self, price=None, port=None, price_emb=None, port_emb=None):
        assert price is not None or price_emb is not None, "Either price or price_emb must be passed"
        assert port is not None or port_emb is not None, "Either port or port_emb must be passed"

        price_emb = price_emb if price_emb is not None else self.get_price_emb(price)
        port_emb = port_emb if port_emb is not None else self.get_port_emb(port)
        return torch.cat([price_emb, port_emb], dim=-1)
        # return price_emb

    def get_price_emb(self, price):
        # price_emb = self.conv_layers(torch.log(price))
        price_emb = self.conv_layers(price)
        return self.act(self.project(price_emb.view(price_emb.shape[0], -1)))

    def get_port_emb(self, port):
        return self.port_embed(port)

class _ConvModel_Test(Model):
    def __init__(self, *, min_tf, n_assets, action_atoms, d_model=256, channels=[32, 64, 64], kernels=[3, 3, 3],
                 strides=[1, 1, 1], act=nn.ReLU, **params):
        super().__init__()
        assert len(kernels) == len(strides) == len(channels)
        self.action_atoms = action_atoms
        self.d_model = d_model
        self.n_assets = n_assets
        self.c1 = nn.Conv1d(n_assets, 32, 3, 1)
        self.c2 = nn.Conv1d(32, 64, 3, 1)
        self.c3 = nn.Conv1d(64, 64, 3, 1)
        example = torch.randn(1, n_assets, min_tf)
        out_shape = self.c3(self.c2(self.c1(example))).shape
        self.fc = nn.Linear(out_shape[1] * out_shape[2], d_model)
        self.out = nn.Linear(d_model, n_assets*action_atoms)
        self.act = nn.ReLU()


    def forward(self, x):
        x = self.act(self.c1(x.price.transpose(1, -1)))
        x = self.act(self.c2(x))
        x = self.act(self.c3(x))
        x = self.act(self.fc(x.view(x.shape[0], -1)))
        x = self.out(x).view(x.shape[0], self.n_assets, self.action_atoms)
        return x

