__all__ = ["load_config", "save_config", "default_device",
           "save_to_hdf", "load_from_hdf",
           "State", "SARSD", "ReplayBuffer",
           "Config", "make_config",
           "DiscreteActionSpace"]
import time
import sys
from pathlib import Path
from typing import Union, Iterable
from abc import ABC, abstractmethod

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

####################################################################################
################################      GENERAL     ##################################
####################################################################################
def time_profile(repeats, out_results=False, **kwargs):
    """
    Time functions by passing callables as keyword arguments
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


####################################################################################
################################      AGENT       ##################################
####################################################################################
class ActionSpace(ABC):

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def shape(self):
        pass

class DiscreteRangeSpace(ActionSpace):
    def __init__(self, ranges: tuple, n: int):
        assert len(ranges) == 2
        self.ranges = ranges
        self.low = ranges[0]
        self.high = ranges[1]
        self.n = n
        self.action_multiplier = 1

    def sample(self):
        action = np.random.randint(self.low, self.high+1, self.n)
        return action * self.action_multiplier

    @property
    def shape(self):
        return (self.n, )

class DiscreteActionSpace(ActionSpace):
    def __init__(self, actions: Iterable, probs: Iterable=None, n: int=1):
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
def calc_conv_out_shape(in_shape, layers):
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
def list_2_dict(list_of_dicts: list):
    """
    aggregates a list of dicts (all with same keys) into a dict of lists

    the train_loop generator yield dictionaries of metrics at each iteration.
    this allows the loop to be interoperable in different scenarios
    The expense of getting a dict (instead of directly appending to list)
    is probably not too much but come back and profile


    """
    if isinstance(list_of_dicts, dict):
        return list_of_dicts
    if list_of_dicts is not None and len(list_of_dicts) > 0:
        if isinstance(list_of_dicts[0], dict):
            dict_of_lists = {k: [metric[k] for metric in list_of_dicts]
                             for k in list_of_dicts[0].keys()}
            return dict_of_lists
    else:
        return {}

def reduce_train_metrics(metrics: Union[dict, pd.DataFrame], columns: list):
    """
    Takes dict (I.e from list_2_dict) or pandas df
    returns dict/df depending on input type
    """
    if metrics is not None and len(metrics):
        _metrics = type(metrics)() # Create an empty dict or pd df
        for col in metrics.keys():
            if col in columns:
                if isinstance(metrics[col][0], (np.ndarray, torch.Tensor)):
                    _metrics[col] = [m.mean().item() for m in metrics[col]]
                elif isinstance(metrics[col][0], list):
                    _metrics[col] = [np.mean(m).item() for m in metrics[col]]
                else:
                    _metrics[col] = metrics[col]
            else:
                _metrics[col] = metrics[col] # Copy might be needed for numpy arrays / torch tensors
    else:
        return metrics
    return _metrics

def reduce_test_metrics(test_metrics, cols=('returns', 'equity', 'cash', 'margin')):
    out = []
    if isinstance(test_metrics, dict):
        return list_2_dict(reduce_test_metrics([test_metrics], cols=cols))
    keys = test_metrics[0].keys()
    for m in test_metrics:
        _m = {}
        for k in keys:
            if k not in cols:
                _m[k] = m[k]
            else:
                if isinstance(m[k], (np.ndarray, torch.Tensor)):
                    _m[k] = m[k].mean().item()
                elif isinstance(m[k], list):
                    _m[k] = np.mean(m[k])
                else:
                    try:
                        _m[k] = np.mean(m[k])
                    except Exception as E:
                        import traceback
                        traceback.print_exc()
                        print("col passed to reduce_test_metrics did not contain ndarray/list/tensor")
                        print("np.mean tried anyway and failed")
        out.append(_m)
    return out
