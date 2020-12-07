from typing import Union, List, Iterable
import math

import torch
import torch.nn as nn


def make_conv1d_layers(input_shape,
                       kernels,
                       channels,
                       strides=None,
                       preserve_window_len=False,
                       act=nn.GELU,
                       causal_dim=0):
    """
    For Vanilla Conv Layers used in conv_net - No dilation etc
    """
    if strides is None:
        strides = [1 for i in range(len(kernels))]
    assert len(kernels) == len(strides) == len(channels)
    assert len(input_shape) == 2
    window_len = input_shape[0]
    input_feats = input_shape[1]
    channels = [input_feats] + channels
    conv_layers = []
    for i, kernel in enumerate(kernels):
        conv = nn.Conv1d(channels[i],
                         channels[i + 1],
                         kernel,
                         stride=strides[i])
        if preserve_window_len:
            arb_input = (window_len, )
            # CAUSAL_DIM=0 assumes 0 is time dimension for input to calc_pad
            causal_pad = calc_pad_to_conserve(arb_input,
                                              conv,
                                              causal_dim=causal_dim)
            conv_layers.append(nn.ReplicationPad1d(causal_pad))
        conv_layers.append(conv)
        conv_layers.append(act())
    return nn.Sequential(*conv_layers)

@torch.no_grad()
def xavier_initialization(m, linear_range=(-3e-3, 3e-3)):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, linear_range[0], linear_range[1])
    elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        scale = 1 / math.sqrt(fan_in)
        nn.init.uniform_(m.weight, -scale, scale)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


@torch.no_grad()
def orthogonal_initialization(m, gain=1.):
    """
    From 'Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks. Saxe et al (2013)'.
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def calc_conv_out_shape(in_shape: Union[tuple, int], layers: List[nn.Module]):
    """
    Calculates output shape of input_shape going through the given conv layers
    in_shape: (H, W)
    layers: list of pytorch convolution layers
    """
    shape = in_shape
    padding_classes2d = (nn.ConstantPad2d, nn.ReflectionPad2d,
                         nn.ReplicationPad2d, nn.ZeroPad2d)
    padding_classes1d = (nn.ConstantPad1d, nn.ReflectionPad1d,
                         nn.ReplicationPad1d)
    if isinstance(shape, Iterable):
        if len(shape) == 2:
            for layer in layers:
                if isinstance(layer, nn.Conv2d):
                    h_out = (
                        (shape[0] + 2 * layer.padding[0] - layer.dilation[0] *
                         (layer.kernel_size[0] - 1) - 1) / layer.stride[0]) + 1
                    w_out = (
                        (shape[1] + 2 * layer.padding[1] - layer.dilation[1] *
                         (layer.kernel_size[1] - 1) - 1) / layer.stride[1]) + 1
                    shape = (int(h_out), int(w_out))
                elif isinstance(layer, padding_classes2d):
                    h_out = shape[0] + layer.padding[0] + layer.padding[1]
                    w_out = shape[1] + layer.padding[2] + layer.padding[3]
                    shape = (int(h_out), int(w_out))
        elif len(shape) == 1:
            for layer in layers:
                if isinstance(layer, nn.Conv1d):
                    out = (
                        (shape[0] + 2 * layer.padding[0] - layer.dilation[0] *
                         (layer.kernel_size[0] - 1) - 1) / layer.stride[0]) + 1
                    shape = (int(out), )
                elif isinstance(layer, padding_classes1d):
                    out = shape[0] + layer.padding[0] + layer.padding[1]
                    shape = (int(out), )
    elif isinstance(shape, int):
        return calc_conv_out_shape((shape, ), layers)
    else:
        raise ValueError("in_shape must be an iterable or int (for conv1d)")
    return shape


def calc_pad_to_conserve(in_shape: Union[int, tuple],
                         layer: nn.Module,
                         causal_dim=0) -> tuple:
    """
    Outputs the required padding on the input to conserve its shape, after passing
    through the given convolutional layers.
    If in_shape is 2d and layers are Conv2D, output tuple is (H_up, H_down, W_left, W_right)

    #### Important for causal dim
    If causal_dim is specified, the padding is assymmetric for that dim of the shape
    and the padding is applied to the first element of the pair for that dimenion
    I.e
    in_shape = (64, 64), layer.kernel = (3, 3)
    with asymmetric causal padding (dim 0):
    to_pad = (2, 0, 1, 1)
    """
    out_shape = calc_conv_out_shape(in_shape, [
        layer,
    ])
    pad = []
    if isinstance(in_shape, Iterable):
        if len(in_shape) == 2:
            for i, _ in enumerate(in_shape):
                diff = (in_shape[i] - out_shape[i]) / 2
                if causal_dim == i:
                    pad += [int(diff * 2), 0]
                else:
                    pad += [int(diff), int(diff)]
            post_pad_input_shape = (pad[0] + pad[1] + in_shape[0],
                                    pad[2] + pad[3] + in_shape[1])
            post_pad_output_shape = calc_conv_out_shape(
                post_pad_input_shape, [
                    layer,
                ])
        elif len(in_shape) == 1:
            diff = (in_shape[0] - out_shape[0]) / 2
            if causal_dim == 0:
                pad += [int(diff * 2), 0]
            else:
                pad += [int(diff), int(diff)]
            post_pad_input_shape = (pad[0] + pad[1] + in_shape[0], )
        post_pad_output_shape = calc_conv_out_shape(post_pad_input_shape, [
            layer,
        ])
    elif isinstance(in_shape, int):
        return calc_pad_to_conserve((in_shape, ), layer, causal_dim=causal_dim)
    else:
        raise ValueError("in_shape must be an iterable or int")
    # TEST HERE THAT THE PADS CONSERVE INPUT SHAPE
    assert tuple(in_shape) == tuple(post_pad_output_shape), \
        "padding calc unsuccessful in conserving input shape." +\
        f" in= {in_shape}, padded={post_pad_output_shape} "
    return tuple(pad)
