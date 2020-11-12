from collections import deque
import math
from random import sample
import numpy as np
from .data import SARSD, State

class ReplayBuffer:
    """
    Experience Replay Buffer generalized for n-step returns
    """
    def __init__(self, size, nstep_return, discount):
        self.size = size
        self.nstep_return = nstep_return
        self.discount = discount
        self.discounts = [math.pow(self.discount, i) for i in range(nstep_return)]
        self._buffer = [None] * size
        # For Small lists, pop(0) has similar performance to deque().popleft()
        # And better performance for iteration when calculation the discounted sum
        # self._nstep_buffer = deque(maxlen=nstep)
        self._nstep_buffer = []
        self.filled = 0
        self.current_idx = 0

    @classmethod
    def from_config(cls, config):
        return cls(config.rb_size, config.nstep_return, config.agent_config.discount)

    @property
    def buffer(self):
        return self._buffer

    def get_nstep_sarsd(self):
        """
        Calculates nstep discounted return from the nstep buffer
        and returns the sarsd with the adjusted return and next_state offset to t+n
        """
        reward = sum([self.discounts[i]*dat.reward for i, dat in
                      enumerate(self._nstep_buffer)])
        nstep_sarsd = self._nstep_buffer.pop(0)
        nstep_sarsd.reward = reward
        if len(self._nstep_buffer):
            nstep_sarsd.next_state = self._nstep_buffer[-1].next_state
        return nstep_sarsd

    def add(self, sarsd):
        """
        Adds sarsd to nstep buffer.
        If nstep buffer is full, adds to replay buffer first
        """
        if len(self._nstep_buffer) == self.nstep_return:
            nstep_sarsd = self.get_nstep_sarsd()
            self._add_to_replay(nstep_sarsd)
        self._nstep_buffer.append(sarsd)
        if sarsd.done:
            while len(self._nstep_buffer) > 0:
                nstep_sarsd = self.get_nstep_sarsd()
                self._add_to_replay(nstep_sarsd)

    def _add_to_replay(self, sarsd):
        """
        Adds the given sarsd (assuming adjusted for nstep returns)
        to  the replay buffer
        """
        self._buffer[self.current_idx] = sarsd
        self.current_idx = (self.current_idx + 1) % self.size
        if self.filled < self.size:
            self.filled += 1

    def sample(self, n):
        if self.filled < self.size:
            return self.batchify(sample(self._buffer[:self.filled], n))
        else:
            return self.batchify(sample(self._buffer, n))

    def batchify(self, sample):
        try:
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
        except:
            import traceback; traceback.print_exc()
            import ipdb; ipdb.set_trace()
        return SARSD(state, action, reward, next_state, done)

    def get_full(self):
        return self.batchify(self._buffer[:self.filled])

    def get_latest(self, size):
        return self.batchify(self._buffer[self.filled-size: self.filled])

    def clear(self):
        self._buffer = [None] * self.size
        self.filled = 0
        self.current_idx = 0
        self._nstep_buffer = []

    def flush_nstep_buffer(self):
        """
        """
        pass

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._buffer[item]
        elif isinstance(item, slice):
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


