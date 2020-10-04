from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from ..utils.data import State
from ..cpp import State as StateA


class Preprocessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def stream(self, data):
        pass

    @abstractmethod
    def currentData(self):
        pass


class WindowedStacker(Preprocessor):
    def __init__(self, window_len):
        self.k = window_len
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
        self.price_buffer.append(state.price)
        self.portfolio_buffer.append(state.portfolio)
        self.time_buffer.append(state.timestamp)

    def stream(self, data):
        if isinstance(data, tuple): # assume srdi
            self.stream_srdi(data)
        elif isinstance(data, (StateA, State)):
            self.stream_state(data)

    def current_data(self):
        return State(np.array(self.price_buffer),
                     self.portfolio_buffer[-1],
                     self.time_buffer[-1])



