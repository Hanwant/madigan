from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    price: np.ndarray
    portfolio: np.ndarray
    timestamp: int


@dataclass
class SARSD:
    state: State
    action: np.ndarray
    reward: float
    next_state: State
    done: bool
