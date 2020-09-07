from random import sample
import numpy as np
from .data import SARSD, State

class ReplayBuffer:

    def __init__(self, size):
        self.size = size
        self._buffer = [None] * size
        self.filled = 0
        self.current_idx = 0

    @property
    def buffer(self):
        return self._buffer

    def add(self, sarsd):
        self._buffer[self.current_idx] = sarsd
        self.current_idx = (self.current_idx + 1) % self.size
        if self.filled < self.size:
            self.filled += 1

    def sample(self, n):
        if self.filled < self.size:
            return self.batchify(sample(self._buffer[:self.filled], n))
        else:
            return self.batchify(sample(self._buffer, n))

    def _batchify(self, sample):
        state_price = np.array([s.state.price for s in sample])
        state_port = np.array([s.state.port for s in sample])
        state = State(state_price, state_port)
        next_state_price = np.array([s.next_state.price for s in sample])
        next_state_port = np.array([s.next_state.port for s in sample])
        next_state = State(next_state_price, next_state_port)
        action = np.array([s.action for s in sample])
        reward = np.array([s.reward for s in sample])
        done = np.array([s.done for s in sample])
        return SARSD(state, action, reward, next_state, done)

    def batchify(self, sample):
        state_price = np.stack([s.state.price for s in sample])
        state_port = np.stack([s.state.port for s in sample])
        state = State(state_price, state_port)
        next_state_price = np.stack([s.next_state.price for s in sample])
        next_state_port = np.stack([s.next_state.port for s in sample])
        next_state = State(next_state_price, next_state_port)
        action = np.array([s.action for s in sample])
        reward = np.array([s.reward for s in sample])
        done = np.array([s.done for s in sample])
        return SARSD(state, action, reward, next_state, done)

    def __getitem__(self, item):
        return self._buffer[item]

    def __len__(self):
        return self.filled

    def __repr__(self):
        return f'replay_buffer size {self.size} filled {self.filled}\n' + repr(self._buffer[:1]).strip(']') + '  ...  ' +repr(self._buffer[-1:]).strip('[')

