from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    price: np.ndarray
    port: np.ndarray


@dataclass
class SARSD:
    state: State
    action: np.ndarray
    reward: float
    next_state: State
    done: bool
