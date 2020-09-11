__all__ = ["load_json", "save_json", "default_device",
           "save_to_hdf", "load_from_hdf",
           "State", "SARSD", "ReplayBuffer"
           "Config", "make_config"]
import time
import sys
from pathlib import Path
import numpy as np
import torch
# from .utils import *
from .config import *
from .data import *
from .replay_buffer import *
from .logging import *


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

def make_grid(n):
    """
    utility function
    makes evenish 2d grid of size n
    useful for arranging plots
    """
    half = int(np.sqrt(n))
    nrows = half
    ncols = int(np.ceil(n/nrows))
    return nrows, ncols

def batchify_sarsd(sarsd):
    sarsd.state.price = sarsd.state.price[None, ...]
    sarsd.state.port = sarsd.state.port[None, ...]
    sarsd.action = sarsd.action[None, ...]
    sarsd.reward = np.array(sarsd.reward)[None, ...]
    sarsd.next_state.price = sarsd.next_state.price[None, ...]
    sarsd.next_state.port = sarsd.next_state.port[None, ...]
    sarsd.done = np.array(sarsd.done)[None, ...]
    return sarsd
