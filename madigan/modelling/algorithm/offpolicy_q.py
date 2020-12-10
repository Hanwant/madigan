import pickle
from typing import Tuple, Union
from abc import abstractmethod
from random import random

import numpy as np
import torch

from .base import Agent
from ...environments import get_env_info
from ...utils.replay_buffer import ReplayBuffer
from ...utils.data import SARSD, State
from ...utils import list_2_dict, DiscreteRangeSpace


class OffPolicyQ(Agent):
    """
    Base class for all off policy agents with experience replay buffers
    """
    def __init__(self, env, preprocessor, input_shape, action_space, discount,
                 nstep_return, replay_size, replay_min_size, noisy_net, eps,
                 eps_decay, eps_min, batch_size, test_steps, unit_size,
                 savepath):
        super().__init__(env, preprocessor, input_shape, action_space,
                         discount, nstep_return, savepath)
        self.noisy_net = noisy_net
        self.eps = eps
        self.eps_decay = max(eps_decay, 1 -
                             eps_decay)  # to make sure we use 0.99999 not 1e-5
        self.eps_min = eps_min
        self.replay_size = replay_size
        self.nstep_return = nstep_return
        self.discount = discount
        self.buffer = ReplayBuffer.from_agent(self)
        self.bufferpath = self.savepath.parent / 'replay.pkl'
        self.replay_min_size = replay_min_size
        self.batch_size = batch_size
        self.test_steps = test_steps
        self.unit_size = unit_size
        self.action_atoms = self.action_space.action_atoms
        self.n_assets = self.action_space.n
        self.centered_actions = np.arange(
            self.action_atoms) - self.action_atoms // 2
        self.log_freq = 5000

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
    def action_to_transaction(self, action: Union[np.ndarray, torch.tensor]):
        pass

    def reset_state(self) -> State:
        state = self._env.reset()
        self._preprocessor.reset_state()
        self._preprocessor.stream_state(state)
        self._preprocessor.initialize_history(self._env)
        return self._preprocessor.current_data()

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
        self.model_b.train()
        self.model_t.train()
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
            self.model_b.sample_noise()
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
                _trn_metrics['eps'] = self.eps
                _trn_metrics['running_reward'] = running_reward
                trn_metrics.append(_trn_metrics)
                self.training_steps += 1

                if self.training_steps % log_freq == 0:
                    yield trn_metrics
                    trn_metrics.clear()

                if self.training_steps > max_steps:
                    yield trn_metrics
                    break

            self.env_steps += 1

    def explore(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        if self.noisy_net:
            action = self.get_action(state, target=False).cpu().numpy()
        else:
            if random() < self.eps:
                action = self.get_action(state, target=False).cpu().numpy()
            else:
                action = self.action_space.sample()
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return action, self.action_to_transaction(action)

    @torch.no_grad()
    def test_episode(self, test_steps=None, reset=True, target=True) -> dict:
        self.model_b.eval()
        self.model_t.eval()
        test_steps = test_steps or self.test_steps
        if reset:
            self.reset_state()
        self._preprocessor.initialize_history(
            self.env)  # probably already initialized
        state = self._preprocessor.current_data()
        tst_metrics = []
        i = 0
        while i <= test_steps:
            _tst_metrics = {}
            qvals = self.get_qvals(state, target=target)[0].cpu().numpy()
            action = self.get_action(state, target=target).cpu().numpy()
            transaction = self.action_to_transaction(action)
            state, reward, done, info = self._env.step(transaction)
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