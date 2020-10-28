from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from ..utils.data import State
# from ..environments.cpp import State as StateA

def make_preprocessor(config):
    if config.preprocessor_type in ("WindowedStacker", "StackerDiscrete"):
        return StackerDiscrete.from_config(config)
    elif config.preprocessor_type in ("StackerContinuous"):
        return StackerContinuous.from_config(config)

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


class StackerDiscrete(Preprocessor):
    def __init__(self, window_len):
        self.k = window_len
        self.min_tf = self.k
        self.price_buffer = deque(maxlen=self.k)
        self.portfolio_buffer = deque(maxlen=self.k)
        self.time_buffer = deque(maxlen=self.k)
        self.feature_output_shape = (self.k, 1)

    @classmethod
    def from_config(cls, config):
        window_len = config.preprocessor_config.window_length
        return cls(window_len)

    def __len__(self):
        return len(self.price_buffer)

    def stream_state(self, state):
        self.price_buffer.append(np.array(state.price, copy=True))
        self.portfolio_buffer.append(np.array(state.portfolio, copy=True))
        self.time_buffer.append(np.array(state.timestamp, copy=True))

    def stream(self, data):
        if isinstance(data, tuple):
            self.stream_state(data[0]) # assume srdi
        else: # assume data is State
            self.stream_state(data)

    def current_data(self):
        return State(np.array(self.price_buffer, copy=True),
                     np.array(self.portfolio_buffer, copy=True),
                     np.array(self.time_buffer, copy=True))

    def initialize_history(self, env):
        while len(self) < self.k:
            _state, reward, done, info = env.step()
            self.stream_state(_state)

    def reset_state(self):
        self.price_buffer.clear()
        self.portfolio_buffer.clear()
        self.time_buffer.clear()

class StackerContinuous(StackerDiscrete):
    def __init__(self, window_len):
        self.timeframe = window_len
        self.min_tf = self.k
        self.price_buffer = deque()
        self.portfolio_buffer = deque()
        self.time_buffer = deque()
        self.current_size = 0

    def stream_state(self, state):
        self.price_buffer.append(np.array(state.price, copy=True))
        self.portfolio_buffer.append(np.array(state.portfolio, copy=True))
        self.time_buffer.append(np.array(state.timestamp, copy=True))
        self.current_size += 1
        if self.current_size > 1:
            # Test difference in real timestamps
            # Generalized to indexes instead of actual timestamps
            while (self.time_buffer[-1] - self.time_buffer[0]) > self.timeframe:
                self.portfolio_buffer.popleft()
                self.price_buffer.popleft()
                self.time_buffer.popleft()

