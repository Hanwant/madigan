import time
import sys
from pathlib import Path
import numpy as np
import torch


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

def batchify_sarsd(sarsd):
    sarsd.state.price = sarsd.state.price[None, ...]
    sarsd.state.port = sarsd.state.port[None, ...]
    sarsd.action = sarsd.action[None, ...]
    sarsd.reward = np.array(sarsd.reward)[None, ...]
    sarsd.next_state.price = sarsd.next_state.price[None, ...]
    sarsd.next_state.port = sarsd.next_state.port[None, ...]
    sarsd.done = np.array(sarsd.done)[None, ...]
    return sarsd

