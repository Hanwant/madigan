import pickle
import math
from collections import deque
from random import sample
from typing import List, Union

import numpy as np
import torch
import cpprb

from .data import SARSD, SARSDH, State

# DQNTYPES refers to agents which share the same obs types
# this includes DDPG as it doesn't store logp
DQNTYPES = ("DQN", "IQN", "DQNReverser", "DQNReverser", "DDPG")


def make_env_dict_cpprb(agent_class, obs_shape, action_shape):
    if agent_class in DQNTYPES:
        env_dict = {
            "state_price": {
                "shape": obs_shape
            },
            "state_portfolio": {
                "shape": action_shape
            },
            "action": {
                "shape": action_shape
            },
            "next_state_price": {
                "shape": obs_shape
            },
            "next_state_portfolio": {
                "shape": action_shape
            },
            "reward": {},
            "done": {},
            "discounts": {}
        }
    else:
        raise NotImplementedError("cpprb spec has been implemented only for "
                                  "the following agent_classes: "
                                  f"{DQNTYPES}")
    return env_dict


class ReplayBufferC:
    """
    Acts as a base class and a factory,
    Its children are wrappers for an rb from the cpprb library
    cls.from_agent method returns an instance of the following:
    - ReplayBufferC_SARSD - for agents in DQNTYPES I.e only needs sarsd

    """

    def add(self, *kw):
        raise NotImplementedError()

    def sample(self, size):
        raise NotImplementedError()

    def get_all_transitions(self):
        raise NotImplementedError()

    @classmethod
    def from_agent(cls, agent):
        agent_type = type(agent).__name__
        nstep_return = agent.nstep_return
        discount = agent.discount
        obs_shape = agent.input_shape
        env_dict = make_env_dict_cpprb(agent_type, obs_shape,
                                       agent.action_space.shape)
        if agent_type in DQNTYPES:
            return ReplayBufferC_SARSD(
                agent.replay_size,
                env_dict=env_dict,
                Nstep={
                    "size": nstep_return,
                    "gamma": discount,
                    "rew": "reward",
                    "next": ["next_state_price", "next_state_port"]
                })
        raise NotImplementedError("cpprb wrapper has been implemented " +
                                  "only for the following agent_classes: " +
                                  f"{DQNTYPES}")


class ReplayBufferC_SARSD(cpprb.ReplayBuffer, ReplayBufferC):
    """
    Replay Buffer wrapper for cpprb for agents in DQNTYPES.
    Wraps replay buffer from cpprb to maintain consistent api
    Uses SARSD and State dataclasses as intermediaries for i/o
    when sampling and adding
    """

    def __len__(self):
        return self.get_stored_size()

    def sample(self, size):
        data = super().sample(size)
        sarsd = SARSD(
            State(data['state_price'],
                  data['state_portfolio']), data['action'], data['reward'],
            State(data['next_state_price'], data['next_state_portfolio']),
            data['done'])
        return sarsd

    def add(self, sarsd):
        super().add(state_price=sarsd.state.price,
                    state_portfolio=sarsd.state.portfolio,
                    reward=sarsd.reward,
                    next_state_price=sarsd.next_state.price,
                    next_state_portfolio=sarsd.next_state.portfolio,
                    done=sarsd.done)

    def save_to_file(self, savepath):
        """
        Extracts transitions and saves them to file
        Instead of pickling the class - not supported (Cython)
        """
        transitions = self.get_all_transitions()
        with open(savepath, 'wb') as f:
            pickle.dump(transitions, f)

    def load_from_file(self, loadpath):
        if loadpath.is_file():
            with open(loadpath, 'rb') as f:
                transitions = pickle.load(f)
                super().add(transitions)


class NStepBuffer:
    """
    Utility class to prevent code duplicaiton.
    It must however be co-ordinated by a class using it (I.e ReplayBuffer)
    It doesn't have access to the main replay buffer and doesn't care if the
    numbers of sampels in its buffer is > nstep. It is the containers
    responsibility to flush and add the processesd samples to a main buffer.
    """
    def __init__(self, nstep, discount):
        self.nstep = nstep
        self.discount = discount
        self.discounts = [
            math.pow(self.discount, i) for i in range(nstep)
        ]
        self._buffer = []

    def add(self, sarsd: SARSD) -> None:
        self._buffer.append(sarsd)

    def full(self) -> bool:
        return len(self._buffer) >= self.nstep

    def pop_nstep_sarsd(self) -> SARSD:
        """
        Calculates nstep discounted return from the nstep buffer
        and returns the sarsd with the adjusted return and next_state offset to t+n
        """
        reward = sum([
            self.discounts[i] * dat.reward
            for i, dat in enumerate(self._buffer)
        ])
        nstep_sarsd = self._buffer.pop(0)
        nstep_sarsd.reward = reward
        if len(self._buffer) > 0:
            # nstep_idx = min(self.nstep, len(self)) - 1
            nstep_idx = -1
            nstep_sarsd.next_state = self._buffer[nstep_idx].next_state
            nstep_sarsd.done = self._buffer[nstep_idx].done
            # if self._nstep_buffer[-1].done:
            #     nstep_sarsd.done = 1
        return nstep_sarsd

    def flush_nstep_buffer(self):
        """
        Useful to call at end of episodes (I.e if not done.)
        """
        out = []
        while self.full():
            out.append(self.pop_nstep_sarsd())
        return out

    def __len__(self):
        return len(self._buffer)


class ReplayBuffer:
    """
    Experience Replay Buffer generalized for n-step returns
    """
    def __init__(self, size, nstep_return, discount):
        self.size = size
        self.nstep_return = nstep_return
        self.discount = discount
        self._buffer = [None] * size
        self._nstep_buffer = NStepBuffer(self.nstep_return, self.discount)
        self.filled = 0
        self.current_idx = 0

    @classmethod
    def from_agent(cls, agent):
        return cls(agent.replay_size, agent.nstep_return,
                   agent.discount)

    @classmethod
    def from_config(cls, config):
        aconf = config.agent_config
        return cls(aconf.replay_size, config.nstep_return,
                   aconf.discount)

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

    def _add_to_replay(self, sarsd):
        """
        Adds the given sarsd (assuming adjusted for nstep returns)
        to  the replay buffer
        """
        self._buffer[self.current_idx] = sarsd
        self.current_idx = (self.current_idx + 1) % self.size
        if self.filled < self.size:
            self.filled += 1

    def sample(self, n: int):
        if self.filled < self.size:
            return self.batchify(sample(self._buffer[:self.filled], n))
        return self.batchify(sample(self._buffer, n))

    def batchify(self, batch: List[SARSD]):
        state_price = np.stack([s.state.price for s in batch])
        state_port = np.stack([s.state.portfolio for s in batch])
        state_time = np.stack([s.state.timestamp for s in batch])
        state = State(state_price, state_port, state_time)
        next_state_price = np.stack([s.next_state.price for s in batch])
        next_state_port = np.stack(
            [s.next_state.portfolio for s in batch])
        next_state_time = np.stack(
            [s.next_state.timestamp for s in batch])
        next_state = State(next_state_price, next_state_port,
                           next_state_time)
        action = np.stack([s.action for s in batch])
        reward = np.stack([s.reward for s in batch])
        done = np.stack([s.done for s in batch])
        return SARSD(state, action, reward, next_state, done)

    def get_full(self):
        return self.batchify(self._buffer[:self.filled])

    def get_latest(self, size):
        return self.batchify(self._buffer[self.filled - size:self.filled])

    def clear(self):
        self._buffer = [None] * self.size
        self.filled = 0
        self.current_idx = 0
        self._nstep_buffer = []

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


class EpisodeReplayBuffer:
    """
    For use with recurrent agents.
    Uses SARSDH as main data structure unit
    """
    def __init__(self, size: int, episode_len: int, min_episode_len: int,
                 episode_overlap: int, store_hidden: bool,
                 nstep_return: int, discount: float):
        self.size = size
        self.episode_len = episode_len
        self.min_episode_len = min_episode_len
        self.episode_overlap = episode_overlap
        self.store_hidden = store_hidden
        if self.min_episode_len < self.episode_overlap:
            raise ValueError("min_episode_len should be >= episode_overlap")
        self.nstep_return = nstep_return
        self.discount = discount
        self.discounts = [
            math.pow(self.discount, i) for i in range(nstep_return)
        ]
        self._buffer = [None] * size
        self._nstep_buffer = []
        self._episode_buffer = []
        self.filled = 0
        self.current_idx = 0
        self.episode_idx = 0

    @classmethod
    def from_agent(cls, agent):
        return cls(agent.episode_len, agent.episode_burn_in_steps,
                   agent.episode_overlap, agent.store_hidden,
                   agent.replay_size, agent.nstep_return, agent.discount)

    @classmethod
    def from_config(cls, config):
        aconf = config.agent_config
        return cls(aconf.episode_len, aconf.episode_burn_in_steps,
                   aconf.episode_overlap, aconf.store_hidden, aconf.replay_size,
                   aconf.nstep_return, aconf.discount)

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
            self.__dict__.update(loaded.__dict__)


    def pop_nstep_sarsd(self):
        """
        Calculates nstep discounted return from the nstep buffer
        and returns the sarsdh with the adjusted return and next_state offset to t+n
        """
        reward = sum([
            self.discounts[i] * dat.reward
            for i, dat in enumerate(self._nstep_buffer)
        ])
        nstep_sarsd = self._nstep_buffer.pop(0)
        nstep_sarsd.reward = reward
        if len(self._nstep_buffer) > 0:
            nstep_sarsd.next_state = self._nstep_buffer[-1].next_state
            nstep_sarsd.done = self._nstep_buffer[-1].done
            # if self._nstep_buffer[-1].done:
            #     nstep_sarsdh.done = 1
        return nstep_sarsd

    def add(self, sarsd):
        """
        Adds sarsdh to nstep buffer.
        If nstep buffer is full, adds to replay buffer first
        """
        self._nstep_buffer.append(sarsd)
        if sarsd.done:
            while len(self._nstep_buffer) > 0:
                nstep_sarsdh = self.pop_nstep_sarsd()
                self._add_to_episode(nstep_sarsdh)
        elif len(self._nstep_buffer) == self.nstep_return:
            nstep_sarsdh = self.pop_nstep_sarsd()
            self._add_to_episode(nstep_sarsdh)

    def _add_to_episode(self, sarsd):
        if self.episode_idx < self.episode_len:
            self._episode_buffer[self.episode_idx] = sarsd
            self.episode_idx += 1
        elif self.episode_idx == self.episode_len:
            self.flush_episode_to_main()
        # HEURISTIC
        if sarsd.done and self.episode_idx >= self.min_episode_len:
            self.flush_episode_to_main()

    def flush_episode_to_main(self):
        """
        Add episode to replay.
        Sets first 'episode_overlap' number of elements to the last in their
        episode and resets self.episode_idx to this number so that new samples
        get added to the end after the overlap with the previous episode.
        """
        sarsd = self.make_episode_sarsd(
            self._episode_buffer[:self.episode_idx])
        self._add_to_replay(sarsd)
        self._episode_buffer[: self.episode_overlap] = \
            self._episode_buffer[self.episode_idx - self.episode_overlap:]
        self.episode_idx = self.episode_overlap

    def _add_to_replay(self, episode: SARSDH):
        """
        Adds the given sarsdh (assuming adjusted for nstep returns)
        to  the replay buffer
        """
        self._buffer[self.current_idx] = episode
        self.current_idx = (self.current_idx + 1) % self.size
        if self.filled < self.size:
            self.filled += 1

    def make_episode_sarsd(self, episode: List[Union[SARSD, SARSDH]]
                            )-> Union[SARSD, SARSDH]:
        state_price = np.stack([s.state.price for s in episode])
        state_port = np.stack([s.state.portfolio for s in episode])
        state_time = np.stack([s.state.timestamp for s in episode])
        state = State(state_price, state_port, state_time)
        next_state_price = np.stack([s.next_state.price for s in episode])
        next_state_port = np.stack(
            [s.next_state.portfolio for s in episode])
        next_state_time = np.stack(
            [s.next_state.timestamp for s in episode])
        next_state = State(next_state_price, next_state_port,
                           next_state_time)
        action = np.stack([s.action for s in episode])
        reward = np.stack([s.reward for s in episode])
        done = np.stack([s.done for s in episode])
        if self.store_hidden:
            hidden_state = episode[0].hidden_state
            return SARSDH(state, action, reward, next_state, done, hidden_state)
        return SARSD(state, action, reward, next_state, done)

    def sample(self, n):
        if self.filled < self.size:
            return self.batchify(sample(self._buffer[:self.filled], n))
        return self.batchify(sample(self._buffer, n))

    def batchify(self, batch: List[Union[SARSD, SARSDH]]
                 ) -> Union[SARSD, SARSDH]:
        state_price = np.stack([episode.state.price for episode in batch])
        state_port = np.stack([episode.state.portfolio for episode in batch])
        state_time = np.stack([episode.state.timestamp for episode in batch])
        state = State(state_price, state_port, state_time)
        next_state_price = np.stack([episode.next_state.price
                                     for episode in batch])
        next_state_port = np.stack([episode.next_state.portfolio
                                    for episode in batch])
        next_state_time = np.stack([episode.next_state.timestamp
                                    for episode in batch])
        next_state = State(next_state_price, next_state_port,
                           next_state_time)
        action = np.stack([episode.action for episode in batch])
        reward = np.stack([episode.reward for episode in batch])
        done = np.stack([episode.done for episode in batch])
        if self.store_hidden:
            hidden_state = np.stack([episode.hidden_state
                                    for episode in batch])
            return SARSDH(state, action, reward, next_state, done, hidden_state)
        return SARSD(state, action, reward, next_state, done)

    def get_full(self):
        return self.batchify(self._buffer[:self.filled])

    def get_latest(self, size):
        return self.batchify(self._buffer[self.filled - size:self.filled])

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

