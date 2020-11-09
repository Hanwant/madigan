__all__ = ["load_config", "save_config", "default_device",
           "save_to_hdf", "load_from_hdf",
           "State", "SARSD", "ReplayBuffer",
           "Config", "make_config",
           "DiscreteActionSpace"]
import time
import sys
from pathlib import Path
from typing import Union, Iterable, List
from abc import ABC, abstractmethod
from functools import partial

import numba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from .utils import *
from .config import *
from .data import *
from .replay_buffer import *
from .logging import *
from .metrics import *

####################################################################################
################################      GENERAL     ##################################
####################################################################################
def time_profile(repeats: int, out_results=False, **kwargs):
    """
    Time functions by passing callables as keyword arguments
    Timings are printed
    repeats = number of times to call each function
    out_results = whether to return the results of the functions
    If repeats > 1, the most recent results is returned and if number of functions>1
    a dict of results is returned
    """
    times = {}
    out = {}
    # timer = time.perf_counter if sys.platform == 'win32' else time.time
    timer = time.perf_counter
    for name, func in kwargs.items():
        res = None
        start = timer()
        for i in range(repeats):
            res = func()
        times[str(name)] = (timer() - start) / repeats
        out[name] = res
    for name, f in kwargs.items():
        print(name, '  :  ', f'{times[str(name)]:0.10f}', ' (s)')
    if out_results:
        if len(kwargs) == 1:
            return out[name]
        return out


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


###############################################################################
################################      AGENT       #############################
###############################################################################
class ActionSpace(ABC):

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def shape(self):
        pass

class DiscreteRangeSpace(ActionSpace):
    def __init__(self, ranges: Union[tuple, int], n: int = 1):
        """
        ranges: tuple [low, high) -> inclusive of low, exclusive of high
        n: int -> I.e number of assets, 1st dim of sample
        """
        if isinstance(ranges, int):
            ranges = (0, ranges)
        assert len(ranges) == 2
        self.ranges = ranges
        self.n = n
        self.low = ranges[0]
        self.high = ranges[1]
        self.action_atoms = self.high - self.low - 1
        self.action_multiplier = 1

    def sample(self):
        action = np.random.randint(self.low, self.high, self.n)
        return action * self.action_multiplier

    @property
    def shape(self):
        return (self.n, )

class DiscreteActionSpace(ActionSpace):
    def __init__(self, actions: Union[tuple, list], probs: Union[tuple, list]=None, n: int=1):
        # assert len(ranges) == 2
        self.actions = actions
        self.probs = probs
        self.n = n
        self.action_multiplier = 1

    def sample(self, n=None):
        n = n or self.n
        action = np.random.choice(self.actions, size=self.n, p=self.probs)
        return action * self.action_multiplier

    @property
    def shape(self):
        return (self.n, )

class ContinuousActionSpace(ActionSpace):
    def __init__(self, low: float, high: float, num_assets, num_actions,
                 transform=lambda x: x):
        self.output_shape = (num_assets, num_actions)
        self.low = low
        self.high = high
        self.dist = partial(np.random.uniform, low, high,
                            size=self.output_shape)
        self.transform = transform

    def sample(self, shape=None):
        if shape is not None:
            return np.random.uniform(self.low, self.high, shape)
        return self.transform(self.dist())

    @property
    def shape(self):
        return self.output_shape

@numba.vectorize([numba.float32(numba.float32), numba.float64(numba.float64)])
def ternarize_array(val):
    """
    for all elements in an ndarray, set elements to their sign
    -1. for neg, 1. for pos, 0. if val if 0.
    """
    if val < 0:
        out = -1.
    elif val > 0.:
        out = 1.
    else:
        out = 0.
    return out

####################################################################################
################################      NETS        ##################################
####################################################################################
def calc_conv_out_shape(in_shape: Union[tuple, int], layers: List[nn.Module]):
    """
    Calculates output shape of input_shape going through a list of pytorch convolutional layers
    in_shape: (H, W)
    layers: list of convolution layers
    """
    shape = in_shape
    padding_classes2d = (nn.ConstantPad2d, nn.ReflectionPad2d, nn.ReplicationPad2d, nn.ZeroPad2d)
    padding_classes1d = (nn.ConstantPad1d, nn.ReflectionPad1d, nn.ReplicationPad1d)
    if isinstance(shape, Iterable):
        if len(shape) == 2:
            for layer in layers:
                if isinstance(layer, nn.Conv2d):
                    h_out = ((shape[0] + 2*layer.padding[0] - layer.dilation[0] *
                              (layer.kernel_size[0] - 1)-1) / layer.stride[0])+1
                    w_out = ((shape[1] + 2*layer.padding[1] - layer.dilation[1] *
                              (layer.kernel_size[1] - 1)-1) / layer.stride[1])+1
                    shape = (int(h_out), int(w_out))
                elif isinstance(layer, padding_classes2d):
                    h_out = shape[0] + layer.padding[0] + layer.padding[1]
                    w_out = shape[1] + layer.padding[2] + layer.padding[3]
                    shape = (int(h_out), int(w_out))
        elif len(shape) == 1:
            for layer in layers:
                if isinstance(layer, nn.Conv1d):
                    out = ((shape[0] + 2*layer.padding[0] - layer.dilation[0] *
                            (layer.kernel_size[0] - 1)-1) / layer.stride[0])+1
                    shape = (int(out),)
                elif isinstance(layer, padding_classes1d):
                    out = shape[0] + layer.padding[0] + layer.padding[1]
                    shape = (int(out),)
    elif isinstance(shape, int):
        return calc_conv_out_shape((shape, ), layers)
    else:
        raise ValueError("in_shape must be an iterable or int (for conv1d)")
    return shape

def calc_pad_to_conserve(in_shape: tuple, layer: nn.Module, causal_dim=0)->tuple:
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
    out_shape = calc_conv_out_shape(in_shape, [layer, ])
    pad = []
    if isinstance(in_shape, Iterable):
        if len(in_shape) == 2:
            for i, _ in enumerate(in_shape):
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
    # TEST HERE THAT THE PADS CONSERVE INPUT SHAPE
    assert tuple(in_shape) == tuple(post_pad_output_shape), \
        "padding calc unsuccessful in conserving input shape." +\
        f" in= {in_shape}, padded={post_pad_output_shape} "
    return tuple(pad)

####################################################################################
################################      TRAINING    ##################################
####################################################################################
def batchify_sarsd(sarsd):
    sarsd.state.price = sarsd.state.price[None, ...]
    sarsd.state.port = sarsd.state.port[None, ...]
    sarsd.action = sarsd.action[None, ...]
    sarsd.reward = np.array(sarsd.reward)[None, ...]
    sarsd.next_state.price = sarsd.next_state.price[None, ...]
    sarsd.next_state.port = sarsd.next_state.port[None, ...]
    sarsd.done = np.array(sarsd.done)[None, ...]
    return sarsd

####################################################################################
###########################      LOGGING  / METRICS     ############################
####################################################################################
