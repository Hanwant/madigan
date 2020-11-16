from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    price: np.ndarray
    portfolio: np.ndarray
    timestamp: int

@dataclass
class Action:
    action: np.ndarray
    transaction: np.ndarray


@dataclass
class SARSD:
    state: State
    action: np.ndarray
    reward: float
    next_state: State
    done: bool
