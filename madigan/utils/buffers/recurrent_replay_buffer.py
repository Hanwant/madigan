import math
from typing import Iterable

import numpy as np
import pickle

from ..data import StateRecurrent, SARSDR


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
