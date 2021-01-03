from typing import Union, List, Iterable
import math

import torch
import torch.nn as nn

ACT_FN_DICT = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'tanhshrink': nn.Tanhshrink,
    'none': lambda: lambda x: x
}


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


@torch.no_grad()
def calc_conv_out_shape(in_shape: Union[tuple, int], layers: List[nn.Module]):
    """
    Returns the output shape of an input going through a given list
    of convolution layers.
    Named calc_conv but generalizes to any layer which changes input shape.
    """
    if isinstance(layers, nn.Module):
        return calc_conv_out_shape(in_shape, [layers])
    if not isinstance(layers, Iterable):
        raise ValueError("layers must be an iterable, containing nn.Modules")

    dummy = torch.randn(*in_shape)
    for layer in layers:
        dummy = layer(dummy)
    return dummy.shape


@torch.no_grad()
def calc_pad_to_conserve1d(in_shape: tuple,
                           layer: nn.Module,
                           causal: bool = False,
                           causal_side: str = 'left') -> tuple:
    """
    Designed for 1D padding assuming the last dim needs padding.
    If causal, the returned tuple will be asymmetric, with the pads
    at pos 0 or 1 for causal_sides 'left' or 'right.
    If not causal and the total number of required pads is an odd number,
    the left side (tuple pos 0) will be 1 larger than the right.

    @params
        in_shape: tuple =  Full shape of input - incl batch dim.
        layer: nn.Module
        causal: bool = whether to pad asymmetrically
        causal_side: str = 'left' or 'right' - side of causal padding.

    """
    if not isinstance(in_shape, tuple):
        raise ValueError("in_shape must be a tuple")
    if causal_side not in ('left', 'right'):
        raise ValueError("causal_side must be either 'left' or 'right")

    dummy = torch.randn(*in_shape)
    if layer(dummy).shape[-1] >= in_shape[-1]:
        return (0, 0)

    pads = 0
    while layer(dummy).shape[-1] < in_shape[-1]:
        pads += 1
        dummy = torch.randn(*in_shape[:-1], in_shape[-1] + pads)

    if causal:
        return (pads, 0) if causal_side == 'left' else (0, pads)
    if pads % 2 == 0:
        return (pads // 2, pads // 2)

    return (pads // 2 + 1, pads // 2)  # if total pad is asymmetric


def _calc_conv_out_shape(in_shape: Union[tuple, int], layers: List[nn.Module]):
    """
    Calculates output shape of input_shape going through the given conv layers
    in_shape: (H, W)
    layers: list of pytorch convolution layers
    """
    if not isinstance(layers, Iterable):
        raise ValueError("layers must be an iterable containing nn.Modules")
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



def _calc_pad_to_conserve(in_shape: Union[int, tuple],
                          layer: nn.Module,
                          causal_dim=0) -> tuple:
    """
    DEPRECATED
    TOO COMPLICATED TO ADD NEW LAYERS, DOING IT BY TRIAL AND ERROR IS MUCH EASIER.

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
        return _calc_pad_to_conserve((in_shape, ),
                                     layer,
                                     causal_dim=causal_dim)
    else:
        raise ValueError("in_shape must be an iterable or int")
    # TEST HERE THAT THE PADS CONSERVE INPUT SHAPE
    assert tuple(in_shape) == tuple(post_pad_output_shape), \
        "padding calc unsuccessful in conserving input shape." +\
        f" in= {in_shape}, padded={post_pad_output_shape} "
    return tuple(pad)
