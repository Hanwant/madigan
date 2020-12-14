import pickle
from typing import Tuple
from abc import abstractmethod

import numpy as np
import torch

from .base import Agent
from ...environments import get_env_info
from ...utils.replay_buffer import ReplayBuffer
from ...utils.data import SARSD, State
from ...utils import list_2_dict, DiscreteRangeSpace


class OffPolicyActorCritic(Agent):
    """
    Base class for all off policy agents with experience replay buffers
    """
    def __init__(self, env, preprocessor, input_shape,
                 action_space, discount, nstep_return, replay_size,
                 replay_min_size, batch_size, test_steps,
                 savepath):
        super().__init__(env, preprocessor, input_shape,
                         action_space, discount, nstep_return, savepath)
        # self.eps = eps
        # self.eps_decay = max(eps_decay, 1 -
        #                      eps_decay)  # to make sure we use 0.99999 not 1e-5
        # self.eps_min = eps_min
        self.replay_size = replay_size
        self.nstep_return = nstep_return
        self.discount = discount
        self.buffer = ReplayBuffer.from_agent(self)
        self.bufferpath = self.savepath.parent / 'replay.pkl'
        self.replay_min_size = replay_min_size
        self.batch_size = batch_size
        self.test_steps = test_steps
        self.n_assets = self.action_space.shape[0]  #  includes cash
        self.log_freq = 10000

    def save_buffer(self):
        self.buffer.save_to_file(self.bufferpath)

    def load_buffer(self):
        self.buffer.load_from_file(self.bufferpath)

    @abstractmethod
    def get_qvals(self, state, target=False):
        pass

    @abstractmethod
    def get_action(self, state, target=False):
        pass

    @abstractmethod
    def action_to_transaction(self, state, target=False):
        pass

    def reset_state(self, random_port_init: bool = True) -> State:
        state = self._env.reset()
        if random_port_init:
            random_port = np.random.uniform(-1, 1, self._env.nAssets)
            random_port = random_port / random_port.sum()
            units = ((random_port[1:] * self._env.equity)
                     / self._env.currentPrices)
            self._env.step(units)
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
        self.train_mode()
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

            if done or self._env.equity > 1e15:
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

                if self.training_steps % log_freq == 0:
                    yield trn_metrics
                    trn_metrics.clear()

                if self.training_steps > max_steps:
                    yield trn_metrics
                    break

            self.env_steps += 1

    @torch.no_grad()
    def test_episode(self, test_steps=None, reset=True, target=True):
        self.test_mode()
        test_steps = test_steps or self.test_steps
        if reset:
            self.reset_state(random_port_init=False)
        self._preprocessor.initialize_history(
            self._env)  # probably already initialized
        state = self._preprocessor.current_data()
        tst_metrics = []
        i = 0
        while i <= test_steps:
            _tst_metrics = {}
            qvals = self.get_qvals(state, target=target)[0].cpu().numpy()
            action, transaction = self.get_action(state, target=target)
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

