from dataclasses import dataclass
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
class SARSDH:
    """
    For use in Recurrent DQN style algorithms
    """
    state: State
    action: np.ndarray
    reward: float
    next_state: State
    done: bool
    hidden: np.ndarray
