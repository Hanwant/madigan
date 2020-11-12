from abc import ABC, abstractmethod
import pickle
from copy import deepcopy
from random import random
from typing import List, Tuple

import numpy as np
import torch

from ...environments import get_env_info
from ...utils.replay_buffer import ReplayBuffer
from ...utils.data import SARSD, State
from ...utils import list_2_dict, DiscreteRangeSpace


class Agent(ABC):
    """
    Base class for all agents
    """
    def __init__(self,
                 env,
                 preprocessor,
                 input_shape: tuple,
                 action_space,
                 discount: float,
                 nstep_return: int,
                 savepath=None):
        self._env = env
        self._preprocessor = preprocessor
        self.input_shape = input_shape
        self._action_space = action_space
        self.discount = discount
        self.nstep_return = nstep_return
        self.savepath = savepath
        self.training_steps = 0
        self.env_steps = 0
        # if not self.savepath.is_dir():
        #     self.savepath.mkdir(parents=True)
        # if (self.savepath/'main.pth').is_file():
        #     self.load_state()

    @property
    def action_space(self):
        return self._action_space

    @property
    def env(self):
        return self._env

    @property
    def preprocessor(self):
        return self._preprocessor

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        pass

    @abstractmethod
    def get_action(self, state: State) -> np.ndarray:
        """
        Fundamental function.
        Returns vector of units to purchase (+/-) for each asset
        """
        pass

    @abstractmethod
    def train_step(self, sarsd=None) -> dict:
        """
        May use provided sarsd or may sample from own internal buffer
        to perform a batched training step
        Returns a dict of metrics I.e {'loss': loss, 'td_error': td_error}
        """
        pass

    @abstractmethod
    def step(self, n: int) -> List[dict]:
        """
        Generator which must be initialized by calling loop = iter(agent.step(n)).
        Fundamental loop in which the agent accumulates experiences via env interaction
        and trains at the designated interval.
        Performs n training steps
        and yields a list of training metrics at the interval specified by config
        """
        pass

    @abstractmethod
    def explore(self, state):
        """
        For interacting with the environment in training phase
        I.e may implement an eps-greedy exploration policy
        """
        pass

    @abstractmethod
    def test_episode(self, test_steps=None, reset=False):
        """
        To test agent performance, the agent interacts with the env
        until either test_steps has been reached or done is returned
        """
        pass

    def __call__(self, *args, **kw):
        return self.get_action(*args, **kw)

    @abstractmethod
    def load_state(self):
        pass

    @abstractmethod
    def save_state(self):
        pass


class OffPolicyQ(Agent):
    """
    Base class for all off policy agents with experience replay buffers
    """
    def __init__(self, env, preprocessor, input_shape, action_space, discount,
                 nstep_return, replay_size, replay_min_size, eps, eps_decay,
                 eps_min, batch_size, test_steps, unit_size, savepath):
        super().__init__(env, preprocessor, input_shape, action_space,
                         discount, nstep_return, savepath)
        self.eps = eps
        self.eps_decay = max(eps_decay, 1 -
                             eps_decay)  # to make sure we use 0.99999 not 1e-5
        self.eps_min = eps_min
        self.buffer = ReplayBuffer(replay_size, nstep_return, discount)
        self.bufferpath = self.savepath.parent/'replay.pkl'
        self.replay_min_size = replay_min_size
        self.batch_size = batch_size
        self.test_steps = test_steps
        self.unit_size = unit_size
        self.action_atoms = self.action_space.action_atoms
        self.n_assets = self.action_space.n
        self.centered_actions = np.arange(
            self.action_atoms) - self.action_atoms // 2
        self.log_freq = 10000

    def save_buffer(self):
        with open(self.bufferpath, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self):
        if self.bufferpath.is_file():
            with open(self.bufferpath, 'rb') as f:
                self.buffer = pickle.load(f)

    @abstractmethod
    def get_qvals(self, state, target=False):
        pass

    @abstractmethod
    def get_action(self, state, target=False):
        pass

    def reset_state(self) -> State:
        state = self._env.reset()
        self._preprocessor.reset_state()
        self._preprocessor.stream_state(state)
        self._preprocessor.initialize_history(self._env)

    def initialize_buffer(self):
        if self.bufferpath.is_file():
            self.load_buffer()
        if len(self.buffer) < self.replay_min_size:
            steps = self.replay_min_size - len(self.buffer)
            self.step(steps)

    def step(self, n, reset=True):
        """
        Performs n steps of interaction with the environment
        Accumulates experiences inside self.buffer (replay buffer for offpolicy)
        performs train_step when conditions are met (replay_min_size)
        """
        trn_metrics = []
        if reset:
            self.reset_state()
        state = self._preprocessor.current_data()
        log_freq = min(self.log_freq, n)
        running_reward = 0.  # for logging
        running_cost = 0.  # for logging
        # i = 0
        max_steps = self.training_steps + n
        while True:
            action = self.explore(state)
            _next_state, reward, done, info = self._env.step(action)

            running_cost += np.sum(info.brokerResponse.transactionCost)
            self._preprocessor.stream_state(_next_state)
            next_state = self._preprocessor.current_data()
            if done:
                reward = -1.
            running_reward += reward
            sarsd = SARSD(state, action, reward, next_state, done)
            self.buffer.add(sarsd)

            # print('step: ', self.env_steps, 'buffer size: ',
            # len(self.buffer), end='\r', flush=True)
            if done:
                self.reset_state()
                state = self._preprocessor.current_data()
                running_reward = 0.
            else:
                state = next_state

            if len(self.buffer) > self.replay_min_size:
                _trn_metrics = self.train_step()
                _trn_metrics['eps'] = self.eps
                _trn_metrics['running_reward'] = running_reward
                trn_metrics.append(_trn_metrics)
                self.training_steps += 1

                if self.env_steps % log_freq == 0:
                    yield trn_metrics
                    trn_metrics.clear()

                if self.env_steps > max_steps:
                    yield trn_metrics
                    break

            self.env_steps += 1

    def explore(self, state) -> np.ndarray:
        if random() < self.eps:
            action = self.action_space.sample()
        else:
            action = self.get_action(state, target=False)
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return action

    @torch.no_grad()
    def test_episode(self, test_steps=None, reset=True):
        test_steps = test_steps or self.test_steps
        if reset:
            self.reset_state()
        self._preprocessor.initialize_history(
            self._env)  # probably already initialized
        state = self._preprocessor.current_data()
        tst_metrics = []
        i = 0
        while i <= test_steps:
            _tst_metrics = {}
            qvals = self.get_qvals(state, target=False)[0].cpu().numpy()
            action = self.get_action(state, target=False)
            state, reward, done, info = self._env.step(action)
            self._preprocessor.stream_state(state)
            state = self._preprocessor.current_data()
            _tst_metrics['qvals'] = qvals
            _tst_metrics['reward'] = reward
            _tst_metrics['transaction'] = info.brokerResponse.transactionUnits
            _tst_metrics[
                'transaction_cost'] = info.brokerResponse.transactionCost
            # _tst_metrics['info'] = info
            tst_metrics.append({**_tst_metrics, **get_env_info(self._env)})
            if done:
                break
            i += 1
        return list_2_dict(tst_metrics)

class OffPolicyActorCritic(Agent):
    """
    Base class for all off policy agents with experience replay buffers
    """
    def __init__(self, env, preprocessor, input_shape, action_space, discount,
                 nstep_return, replay_size, replay_min_size, batch_size,
                 expl_noise_sd, test_steps, savepath):
        super().__init__(env, preprocessor, input_shape, action_space,
                         discount, nstep_return, savepath)
        # self.eps = eps
        # self.eps_decay = max(eps_decay, 1 -
        #                      eps_decay)  # to make sure we use 0.99999 not 1e-5
        # self.eps_min = eps_min
        self.buffer = ReplayBuffer(replay_size, nstep_return, discount)
        self.bufferpath = self.savepath.parent/'replay.pkl'
        self.replay_min_size = replay_min_size
        self.batch_size = batch_size
        self.expl_noise_sd = expl_noise_sd
        self.test_steps = test_steps
        self.n_assets = self.action_space.shape[0]
        self.log_freq = 10000

    def save_buffer(self):
        with open(self.bufferpath, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self):
        if self.bufferpath.is_file():
            with open(self.bufferpath, 'rb') as f:
                self.buffer = pickle.load(f)

    @abstractmethod
    def get_qvals(self, state, target=False):
        pass

    @abstractmethod
    def get_action(self, state, target=False):
        pass

    @abstractmethod
    def action_to_transaction(self, state, target=False):
        pass

    def reset_state(self) -> State:
        state = self._env.reset()
        self._preprocessor.reset_state()
        self._preprocessor.stream_state(state)
        self._preprocessor.initialize_history(self._env)

    def initialize_buffer(self):
        if self.bufferpath.is_file():
            self.load_buffer()
        if len(self.buffer) < self.replay_min_size:
            steps = self.replay_min_size - len(self.buffer)
            self.step(steps)

    def step(self, n, reset=True):
        """
        Performs n steps of interaction with the environment
        Accumulates experiences inside self.buffer (replay buffer for offpolicy)
        performs train_step when conditions are met (replay_min_size)
        """
        trn_metrics = []
        if reset:
            self.reset_state()
        state = self._preprocessor.current_data()
        log_freq = min(self.log_freq, n)
        running_reward = 0.  # for logging
        running_cost = 0.  # for logging
        max_steps = self.training_steps + n
        while True:
            action, transaction = self.explore(state)
            _next_state, reward, done, info = self._env.step(transaction)

            running_cost += np.sum(info.brokerResponse.transactionCost)
            self._preprocessor.stream_state(_next_state)
            next_state = self._preprocessor.current_data()
            if done:
                reward = -1.
            running_reward += reward
            sarsd = SARSD(state, action, reward, next_state, done)
            self.buffer.add(sarsd)

            if done:
                self.reset_state()
                state = self._preprocessor.current_data()
                running_reward = 0.
            else:
                state = next_state

            if len(self.buffer) > self.replay_min_size:
                _trn_metrics = self.train_step()
                _trn_metrics['running_reward'] = running_reward
                trn_metrics.append(_trn_metrics)
                self.training_steps += 1

                if self.env_steps % log_freq == 0:
                    yield trn_metrics
                    trn_metrics.clear()

                if self.env_steps > max_steps:
                    yield trn_metrics
                    break

            self.env_steps += 1

    def explore(self, state, noise_sd: float = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        noise_sd = noise_sd or self.expl_noise_sd
        action = self.get_action(state, target=False).cpu()
        action += noise_sd * torch.randn_like(action)
        return action.numpy()[0], self.action_to_transaction(action)[0]

    @torch.no_grad()
    def test_episode(self, test_steps=None, reset=True):
        test_steps = test_steps or self.test_steps
        if reset:
            self.reset_state()
        self._preprocessor.initialize_history(
            self._env)  # probably already initialized
        state = self._preprocessor.current_data()
        tst_metrics = []
        i = 0
        while i <= test_steps:
            _tst_metrics = {}
            qvals = self.get_qvals(state, target=True)[0].cpu().numpy()
            action, transaction = self.explore(state, noise_sd=0.)
            state, reward, done, info = self._env.step(transaction)
            self._preprocessor.stream_state(state)
            state = self._preprocessor.current_data()
            _tst_metrics['qvals'] = qvals
            _tst_metrics['reward'] = reward
            _tst_metrics['action'] = action
            _tst_metrics['transaction'] = info.brokerResponse.transactionUnits
            _tst_metrics[
                'transaction_cost'] = info.brokerResponse.transactionCost
            # _tst_metrics['info'] = info
            tst_metrics.append({**_tst_metrics, **get_env_info(self._env)})
            # tm = tst_metrics[-1]
            # if tm['equity'] < 0.:
            #     import ipdb; ipdb.set_trace()
            if done:
                break
            i += 1
        return list_2_dict(tst_metrics)
