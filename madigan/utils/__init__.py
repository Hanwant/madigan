__all__ = ["load_config", "save_config", "default_device",
           "save_to_hdf", "load_from_hdf",
           "State", "SARSD", "ReplayBuffer",
           "Config", "make_config",
           "DiscreteActionSpace"]
import time
import sys
from pathlib import Path
from typing import Union, Iterable

import numpy as np
import pandas as pd
import torch
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
class DiscreteRangeSpace:
    def __init__(self, ranges: tuple, n: int):
        assert len(ranges) == 2
        self.ranges = ranges
        self.low = ranges[0]
        self.high = ranges[1]
        self.n = n

    def sample(self):
        action = np.random.randint(self.low, self.high+1, self.n)
        return action

    @property
    def shape(self):
        return (self.n, )

class DiscreteActionSpace:
    def __init__(self, actions: Iterable, probs: Iterable=None, n: int=1):
        # assert len(ranges) == 2
        self.actions=actions
        self.probs = probs
        self.n = n

    def sample(self, n=None):
        n = n or self.n
        action = np.random.choice(self.actions, size=self.n, p=self.probs)
        return action

    @property
    def shape(self):
        return (self.n, )


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
