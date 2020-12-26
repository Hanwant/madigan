from dataclasses import dataclass
from typing import Union, Tuple
import numpy as np
import torch

@dataclass
class State:
    """
    Constitutes the 'state' received from trading env
    """
    price: np.ndarray
    portfolio: np.ndarray
    timestamp: np.ndarray

@dataclass
class StateRecurrent:
    """
    Constitutes the 'state' received from trading env
    For use in Recurrent Agents which are autoregressive.
    Hence previous rewards and actions are also considered to be
    part of the 'State'
    Note that action and reward are from PREVIOUS timesteps,
    hence will not be the same as the reward and action in the enclosing SARSDH
    For state, rewards and actions in SARSDH will be ahead by 1 timestamp.
    For next_state, they will be behind by n_step - 1
    If using stored hidden states, a tuple is assigned to hidden (h0, c0)
    for hidden and cell states respectively.
    """
    price: np.ndarray
    portfolio: np.ndarray
    timestamp: np.ndarray
    action: np.ndarray
    reward: float
    hidden: Union[Tuple[torch.Tensor], None]  # (h0, c0)  or None

@dataclass
class Action:
    action: np.ndarray
    transaction: np.ndarray

@dataclass
class SARSD:
    """
    For use in DQN style algorithms
    """
    state: State
    action: np.ndarray
    reward: float
    next_state: State
    done: bool

@dataclass
class SARSDR:
    """
    For use in Recurrent DQN style algorithms
    """
    state: StateRecurrent
    action: np.ndarray
    reward: float
    next_state: StateRecurrent
    done: bool
