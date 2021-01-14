import pickle
import math
from collections import deque
from random import sample
from typing import List, Union, Iterable

import numpy as np
import torch

from .nstep_buffer import NStepBuffer
from .segment_tree import SumTree, MinTree
from ..data import SARSD, SARSDR, State, StateRecurrent

# DQNTYPES refers to agents which share the same obs types
# this includes DDPG as it doesn't store logp


class ReplayBuffer:
    """
    Experience Replay Buffer generalized for n-step returns
    """
    def __init__(self,
                 size,
                 nstep_return,
                 discount,
                 reward_shaper_config={'reward_shaper': 'sum_default'}):
        self.size = size
        self.nstep_return = nstep_return
        self.discount = discount
        self._buffer = [None] * size
        self._nstep_buffer = NStepBuffer(self.nstep_return, self.discount,
                                         reward_shaper_config)
        self.filled = 0
        self.current_idx = 0

    @classmethod
    def from_agent(cls, agent):
        return cls(agent.replay_size, agent.nstep_return, agent.discount,
                   agent.reward_shaper_config)

    @classmethod
    def from_config(cls, config):
        aconf = config.agent_config
        return cls(aconf.replay_size, aconf.nstep_return, aconf.discount,
                   config.reward_shaper_config)

    @property
    def buffer(self):
        return self._buffer

    def save_to_file(self, savepath):
        with open(savepath, 'wb') as f:
            pickle.dump(self, f)

    def load_from_file(self, loadpath):
        if loadpath.is_file():
            with open(loadpath, 'rb') as f:
                loaded = pickle.load(f)
            self.__dict__ = loaded.__dict__
            if 'cached_idxs' in self.__dict__.keys():
                self.__dict__['cached_idxs'] = None

    def add(self, sarsd):
        """
        Adds sarsd to nstep buffer.
        If nstep buffer is full, adds to replay buffer first
        """
        self._nstep_buffer.add(sarsd)
        if self._nstep_buffer.full():
            nstep_sarsd = self._nstep_buffer.pop_nstep_sarsd()
            self._add_to_replay(nstep_sarsd)
        if sarsd.done:
            while len(self._nstep_buffer) > 0:
                nstep_sarsd = self._nstep_buffer.pop_nstep_sarsd()
                self._add_to_replay(nstep_sarsd)

    def _add_to_replay(self, nstep_sarsd: SARSD):
        """
        Adds the given sarsd (assuming adjusted for nstep returns)
        to  the replay buffer
        """
        self._buffer[self.current_idx] = nstep_sarsd
        self.current_idx = (self.current_idx + 1) % self.size
        if self.filled < self.size:
            self.filled += 1

    def sample(self, n: int):
        """
        Returns a tuple of sampled batch and importance sampling weights.
        As this uniform sampling buffer doesn't provide prioritized replay,
        weights will be None (returned to keep consistent interface)
        """
        idxs = self._sample_idxs(n)
        return self._sample(idxs), None

    def sample_old(self, n: int):
        if self.filled < self.size:
            return self.batchify(sample(self._buffer[:self.filled], n)), None
        return self.batchify(sample(self._buffer, n)), None

    def _sample_idxs(self, n):
        return np.random.randint(0, self.filled, n)

    def _sample(self, idxs):
        """ Given batch indices, returns SARSD of collated samples"""
        try:
            state_price = np.stack([self._buffer[idx].state.price for idx in idxs])
            state_port = np.stack(
                [self._buffer[idx].state.portfolio for idx in idxs])
            state_time = np.stack(
                [self._buffer[idx].state.timestamp for idx in idxs])
            state = State(state_price, state_port, state_time)
            next_state_price = np.stack(
                [self._buffer[idx].next_state.price for idx in idxs])
            next_state_port = np.stack(
                [self._buffer[idx].next_state.portfolio for idx in idxs])
            next_state_time = np.stack(
                [self._buffer[idx].next_state.timestamp for idx in idxs])
            next_state = State(next_state_price, next_state_port, next_state_time)
            action = np.stack([self._buffer[idx].action for idx in idxs])
            reward = np.stack([self._buffer[idx].reward for idx in idxs])
            done = np.stack([self._buffer[idx].done for idx in idxs])
        except:
            import traceback; traceback.print_exc()
            import ipdb; ipdb.set_trace()

        return SARSD(state, action, reward, next_state, done)

    # def batchify(self, batch: List[SARSD]):
    #     state_price = np.stack([s.state.price for s in batch])
    #     state_port = np.stack([s.state.portfolio for s in batch])
    #     state_time = np.stack([s.state.timestamp for s in batch])
    #     state = State(state_price, state_port, state_time)
    #     next_state_price = np.stack([s.next_state.price for s in batch])
    #     next_state_port = np.stack([s.next_state.portfolio for s in batch])
    #     next_state_time = np.stack([s.next_state.timestamp for s in batch])
    #     next_state = State(next_state_price, next_state_port, next_state_time)
    #     action = np.stack([s.action for s in batch])
    #     reward = np.stack([s.reward for s in batch])
    #     done = np.stack([s.done for s in batch])
    #     return SARSD(state, action, reward, next_state, done)

    def get_full(self):
        return self._sample(range(self.filled))

    def get_latest(self, size):
        return self._sample(range(self.filled - size, self.filled))

    def clear(self):
        self._buffer.clear()
        self._buffer = [None] * self.size
        self.filled = 0
        self.current_idx = 0
        self._nstep_buffer.clear()

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._buffer[item]
        if isinstance(item, slice):
            step = item.step if item.step is not None else 1
            return self._sample(range(item.start, item.stop, step))

    def __len__(self):
        return self.filled

    def __repr__(self):
        return f'replay_buffer size {self.size} filled {self.filled}\n' + \
            repr(self._buffer[:1]).strip(']') + '  ...  ' + \
            repr(self._buffer[-1:]).strip('[')
