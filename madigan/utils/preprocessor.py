from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from ..utils.data import State
from ..environments.cpp import State as StateA

def make_preprocessor(config):
    if config['preprocessor_type'] == "WindowedStacker":
        return WindowedStacker(config.preprocessor_config['window_length'])
    else:
        raise NotImplementedError(f"{config['preprocessor_type']} is not implemented ")


class Preprocessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def stream(self, data):
        pass

    @abstractmethod
    def current_data(self):
        pass

    @classmethod
    def from_config(cls, config):
        return make_preprocessor(config)

    @abstractmethod
    def initialize_history(self):
        pass


class WindowedStacker(Preprocessor):
    def __init__(self, window_len):
        self.k = window_len
        self.min_tf = self.k
        self.price_buffer = deque(maxlen=self.k)
        self.portfolio_buffer = deque(maxlen=self.k)
        self.time_buffer = deque(maxlen=self.k)

    def __len__(self):
        return len(self.price_buffer)

    def stream_srdi(self, srdi):
        self.price_buffer.append(srdi[0].price)
        self.portfolio_buffer.append(srdi[0].portfolio)
        self.time_buffer.append(srdi[0].timestamp)

    def stream_state(self, state):
        self.price_buffer.append(np.array(state.price, copy=True))
        self.portfolio_buffer.append(np.array(state.portfolio, copy=True))
        self.time_buffer.append(np.array(state.timestamp, copy=True))

    def stream(self, data):
        if isinstance(data, tuple): # assume srdi
            self.stream_srdi(data)
        elif isinstance(data, (StateA, State)):
            self.stream_state(data)

    def current_data(self):
        return State(np.array(self.price_buffer, copy=True),
                     self.portfolio_buffer[-1],
                     self.time_buffer[-1])

    def initialize_history(self, env):
        while len(self) < self.k:
            _state, reward, done, info = env.step()
            self.stream_state(_state)



