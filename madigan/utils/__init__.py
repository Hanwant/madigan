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
