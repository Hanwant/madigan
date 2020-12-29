import pickle
import math
from collections import deque
from random import sample
from typing import List, Union, Iterable

import numpy as np
import torch
import cpprb

from .segment_tree import SumTree, MinTree
from .data import SARSD, SARSDR, State, StateRecurrent

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
        self.discounts = [math.pow(self.discount, i) for i in range(nstep)]
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
        if agent.prioritized_replay:
            return PrioritizedReplayBuffer(agent.replay_size,
                                           agent.nstep_return, agent.discount,
                                           agent.per_alpha, agent.per_beta,
                                           agent.per_beta_steps)
        return cls(agent.replay_size, agent.nstep_return, agent.discount)

    @classmethod
    def from_config(cls, config):
        aconf = config.agent_config
        if aconf.prioritized_replay:
            return PrioritizedReplayBuffer(aconf.replay_size,
                                           aconf.nstep_return, aconf.discount,
                                           aconf.per_alpha, aconf.per_beta,
                                           aconf.per_beta_steps)

        return cls(aconf.replay_size, aconf.nstep_return, aconf.discount)

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


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self,
                 size,
                 nstep_return,
                 discount,
                 alpha=0.6,
                 beta=0.4,
                 beta_steps=10**5,
                 min_pa=0.,
                 max_pa=1.,
                 eps=0.01):
        super().__init__(size, nstep_return, discount)
        self.alpha = alpha
        self.beta = beta
        self.beta_diff = (1. - beta) / beta_steps
        self.min_pa = min_pa
        self.max_pa = max_pa
        self.eps = eps
        tree_size = 1
        while tree_size < size:
            tree_size <<= 1
        self.tree_sum = SumTree(tree_size)
        self.tree_min = MinTree(tree_size)
        self.cached_idxs = None

    def _add_to_replay(self, nstep_sarsd: SARSD):
        super()._add_to_replay(nstep_sarsd)
        self.tree_min[self.current_idx] = self.max_pa
        self.tree_sum[self.current_idx] = self.max_pa

    def sample(self, n: int):
        assert self.cached_idxs is None, "Update priorities before sampling"

        total_pa = self.tree_sum.reduce(0, self.filled)
        rand = np.random.rand(n) * total_pa
        self.cached_idxs = [self.tree_sum.find_prefixsum_idx(r) for r in rand]
        self.beta = min(1., self.beta + self.beta_diff)

        weight = self._calc_weight(self.cached_idxs)
        batch = self._sample(self.cached_idxs)
        # batch = [self._buffer[idx] for idx in self.cached_idxs]
        return batch, weight

    def _calc_weight(self, idxs: list) -> np.ndarray:
        min_pa = self.tree_min.reduce(0, self.filled)
        weight = [(self.tree_sum[i] / min_pa)**-self.beta for i in idxs]
        weight = np.array(weight, dtype=np.float32)
        return weight

    def update_priority(self, td_error: torch.Tensor):
        assert self.cached_idxs is not None, "Sample another batch before updating priorities"
        assert len(td_error.shape) == 1
        pa = self._calc_pa(td_error).cpu().numpy()
        for i, idx in enumerate(self.cached_idxs):
            self.tree_sum[idx] = pa[i]
            self.tree_min[idx] = pa[i]
        self.cached_idxs = None

    def _calc_pa(self, td_error: torch.Tensor) -> torch.Tensor:
        return torch.clip((td_error + self.eps)**self.alpha, self.min_pa,
                          self.max_pa)


class EpisodeReplayBuffer:
    """
    For use with recurrent agents.
    Uses SARSDR as main data structure unit
    Using old version of ReplayBuffer as template (doesn't inherit)
    so self._nstep_buffer is not abstracted out via NstepBuffer class interface
    """
    def __init__(self, size: int, episode_len: int, min_episode_len: int,
                 episode_overlap: int, store_hidden: bool, nstep_return: int,
                 discount: float):
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
        self._episode_buffer = [None] * episode_len
        self.filled = 0
        self.current_idx = 0
        self.episode_idx = 0

    @classmethod
    def from_agent(cls, agent):
        return cls(agent.replay_size, agent.episode_len, agent.burn_in_steps,
                   agent.episode_overlap, agent.store_hidden,
                   agent.nstep_return, agent.discount)

    @classmethod
    def from_config(cls, config):
        aconf = config.agent_config
        return cls(aconf.replay_size, aconf.episode_len, aconf.burn_in_steps,
                   aconf.episode_overlap, aconf.store_hidden,
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
                nstep_sarsd = self.pop_nstep_sarsd()
                self._add_to_episode(nstep_sarsd)
        elif len(self._nstep_buffer) == self.nstep_return:
            nstep_sarsd = self.pop_nstep_sarsd()
            self._add_to_episode(nstep_sarsd)

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
            self._episode_buffer[len(self._episode_buffer) -
                                 self.episode_overlap:]
        self.episode_idx = self.episode_overlap

    def _add_to_replay(self, episode: SARSDR):
        """
        Adds the given sarsdh (assuming adjusted for nstep returns)
        to  the replay buffer
        """
        self._buffer[self.current_idx] = episode
        self.current_idx = (self.current_idx + 1) % self.size
        if self.filled < self.size:
            self.filled += 1

    def make_episode_sarsd(self, episode: List[SARSDR]) -> SARSDR:
        """ Called when flushing data from Episode BUffer to Main Buffer """
        state_price = np.stack([s.state.price for s in episode])
        state_port = np.stack([s.state.portfolio for s in episode])
        state_time = np.stack([s.state.timestamp for s in episode])
        state_action = np.stack([s.state.action for s in episode])
        state_reward = np.stack([s.state.reward for s in episode])
        hidden = episode[0].state.hidden if self.store_hidden else None
        state = StateRecurrent(state_price, state_port, state_time,
                               state_action, state_reward, hidden)
        next_state_price = np.stack([s.next_state.price for s in episode])
        next_state_port = np.stack([s.next_state.portfolio for s in episode])
        next_state_time = np.stack([s.next_state.timestamp for s in episode])
        next_state_action = np.stack([s.next_state.action for s in episode])
        next_state_reward = np.stack([s.next_state.reward for s in episode])
        next_hidden = episode[0].next_state.hidden \
            if self.store_hidden else None
        state = StateRecurrent(state_price, state_port, state_time,
                               state_action, state_reward, next_hidden)
        next_state = StateRecurrent(next_state_price, next_state_port,
                                    next_state_time, next_state_action,
                                    next_state_reward, next_hidden)
        action = np.stack([s.action for s in episode])
        reward = np.stack([s.reward for s in episode])
        done = np.stack([s.done for s in episode])
        return SARSDR(state, action, reward, next_state, done)

    def sample(self, n: int):
        """
        Returns a tuple of sampled batch and importance sampling weights.
        As this uniform sampling buffer doesn't provide prioritized replay,
        weights will be None (returned to keep consistent interface)
        """
        idxs = self._sample_idxs(n)
        return self._sample(idxs), None

    # def sample_old(self, n):
    #     if self.filled < self.size:
    #         return self.batchify(sample(self._buffer[:self.filled], n))
    #     return self.batchify(sample(self._buffer, n))

    def _sample_idxs(self, n: int):
        return np.random.randint(0, self.filled, n)

    def _sample(self, idxs: Iterable[int]) -> SARSDR:
        """ Batches sequences into shape (bs, seq_len, -1) -1 being variable"""
        state_price = np.stack([self.buffer[idx].state.price for idx in idxs])
        state_port = np.stack(
            [self.buffer[idx].state.portfolio for idx in idxs])
        state_time = np.stack(
            [self.buffer[idx].state.timestamp for idx in idxs])
        state_action = np.stack(
            [self.buffer[idx].state.action for idx in idxs])
        state_reward = np.stack(
            [self.buffer[idx].state.reward for idx in idxs])
        if self.store_hidden:
            state_hidden = [self.buffer[idx].state.hidden for idx in idxs]
        else:
            state_hidden = None
        state = StateRecurrent(state_price, state_port, state_time,
                               state_action, state_reward, state_hidden)
        next_state_price = np.stack(
            [self.buffer[idx].next_state.price for idx in idxs])
        next_state_port = np.stack(
            [self.buffer[idx].next_state.portfolio for idx in idxs])
        next_state_time = np.stack(
            [self.buffer[idx].next_state.timestamp for idx in idxs])
        next_state_action = np.stack(
            [self.buffer[idx].next_state.action for idx in idxs])
        next_state_reward = np.stack(
            [self.buffer[idx].next_state.reward for idx in idxs])
        if self.store_hidden:
            next_state_hidden = [
                self.buffer[idx].next_state.hidden for idx in idxs
            ]
        else:
            next_state_hidden = None
        next_state = StateRecurrent(next_state_price, next_state_port,
                                    next_state_time, next_state_action,
                                    next_state_reward, next_state_hidden)
        action = np.stack([self.buffer[idx].action for idx in idxs])
        reward = np.stack([self.buffer[idx].reward for idx in idxs])
        done = np.stack([self.buffer[idx].done for idx in idxs])
        return SARSDR(state, action, reward, next_state, done)

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
