from collections import deque
from random import sample
import numpy as np
from .data import SARSD, State

class ReplayBuffer:

    def __init__(self, size, nstep=1):
        self.size = size
        self.nstep = nstep
        self._buffer = [None] * size
        self._nstep_buffer = deque(maxlen=nstep)
        self.nstep_return = 0.
        self.filled = 0
        self.current_idx = 0

    @property
    def buffer(self):
        return self._buffer

    def add(self, sarsd):
        if len(self._nstep_buffer) == self.nstep:
            # _reward = sum([dat.reward for dat in self._nstep_buffer])
            nstep_sarsd = self._nstep_buffer.popleft()
            reward = self.nstep_return
            self.nstep_return -= nstep_sarsd.reward
            nstep_sarsd.reward = reward
            self._buffer[self.current_idx] = nstep_sarsd
            self.current_idx = (self.current_idx + 1) % self.size
            if self.filled < self.size:
                self.filled += 1
        self._nstep_buffer.append(sarsd)
        self.nstep_return += sarsd.reward

    def sample(self, n):
        if self.filled < self.size:
            return self.batchify(sample(self._buffer[:self.filled], n))
        else:
            return self.batchify(sample(self._buffer, n))

    def batchify(self, sample):
        state_price = np.stack([s.state.price for s in sample])
        state_port = np.stack([s.state.portfolio for s in sample])
        state_time = np.stack([s.state.timestamp for s in sample])
        state = State(state_price, state_port, state_time)
        next_state_price = np.stack([s.next_state.price for s in sample])
        next_state_port = np.stack([s.next_state.portfolio for s in sample])
        next_state_time = np.stack([s.next_state.timestamp for s in sample])
        next_state = State(next_state_price, next_state_port, next_state_time)
        action = np.stack([s.action for s in sample])
        reward = np.stack([s.reward for s in sample])
        done = np.stack([s.done for s in sample])
        return SARSD(state, action, reward, next_state, done)

    def get_latest(self, size):
        return self.batchify(self._buffer[self.filled-size: self.filled])

    def __getitem__(self, item):
        return self.batchify(self._buffer[item])

    def __len__(self):
        return self.filled

    def __repr__(self):
        return f'replay_buffer size {self.size} filled {self.filled}\n' + \
            repr(self._buffer[:1]).strip(']') + '  ...  ' + \
            repr(self._buffer[-1:]).strip('[')

    # def _batchify(self, sample):
    #     state_price = np.array([s.state.price for s in sample])
    #     state_port = np.array([s.state.portfolio for s in sample])
    #     state_time = np.array([s.state.timestamp for s in sample])
    #     state = State(state_price, state_port, state_time)
    #     next_state_price = np.array([s.next_state.price for s in sample])
    #     next_state_port = np.array([s.next_state.portfolio for s in sample])
    #     next_state_time = np.array([s.next_state.timestamp for s in sample])
    #     next_state = State(next_state_price, next_state_port, next_state_time)
    #     action = np.array([s.action for s in sample])
    #     reward = np.array([s.reward for s in sample])
    #     done = np.array([s.done for s in sample])
    #     return SARSD(state, action, reward, next_state, done)


