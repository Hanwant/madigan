import os
from typing import Union
from collections import OrderedDict
from pathlib import Path
from random import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .iqn import IQN
from .base import Agent
from .utils import discrete_action_to_transaction, abs_port_norm
from ..utils import get_model_class
from ...environments import make_env
from ..net.conv_net import ConvNet
from ..net.mlp_net import MLPNet
from ...utils import default_device, DiscreteActionSpace, DiscreteRangeSpace
from ...utils import ActionSpace
from ...utils.preprocessor import make_preprocessor
from ...utils.config import Config
from ...utils.data import State, SARSD

# p = type('params', (object, ), params)


class IQNController(IQN):
    """
    Heirarchical IQN.
    The actions of this agent route to a set of pretrained agents.
    The resulting action and transactions are sent to the env.
    Hence it acts a 'controller', enabling a kind of
    meta/heirarchical learning.
    Gradients do not flow to the controlled agents.
    The only difference from normal IQN is the action_to_transaction()
    method which routes to a contained agent, otherwise the interface and
    implementation is exactly the same as IQN.

    Currently, agents are assumed to use the same preprocessor and thus it is
    shared. This is more efficient for the time being but must be revised
    if explored further.

    An extra action is added representing the null agent
    no action -> transaction of 0.

    TODO
    Make actions STICKY via either:
    - negative reward for changing from previous action
      - simple except have to feed in previous actions to agent as well.
      - have to tune how much to penalize
      - messes with q values.
    - loss penalty for changing
      - need to extend SARSD to store prev action
      - bit of extra overhead along with feeding previous actions to agent
      - have to tune penalty
      - doesn't mess with q values.
    - action repeat
      - simplest to implement.
      - no extra overhead.
      - no tuning of penalty or messing with q values
      - tuning required for repeat length.
      - not as dynamic as the other options but may be a good inductive bias
        depending on the behvaiour of the timeseries (i.e regime length).

    Might be easiest with Recurrent IQN where actions are to be fed as inputs
    anyway.

    """
    def __init__(self, agents: OrderedDict, *args, **kw):
        """ See IQN __init__ """
        super().__init__(*args, **kw)

        self.agents = agents
        self.agent_idxs = list(self.agents)  # ordered list of agent keys

        if not all(agents[self.agent_idxs[i]].action_space == agents[
                self.agent_idxs[i - 1]].action_space
                   for i in range(1, len(agents))):
            raise Exception("agents must have same action space")

    @classmethod
    def from_config(cls, config):
        """
        Should be the main entry point to construct agent
        For order of args - see IQN constructor
        """
        env = make_env(config)
        # ASSUMING AGENTS HAVE SAME PREPROCESSOR FOR NOW
        preprocessor = make_preprocessor(config, env.nAssets)
        input_shape = preprocessor.feature_output_shape
        agents = OrderedDict({
            name: IQN.from_config(Config.from_path(conf))
            for name, conf in config.agent_configs.items()
        })
        atoms = len(agents) + 1  # +1 for null agent
        action_space = DiscreteRangeSpace((0, atoms), env.nAssets)
        aconf = config.agent_config
        unit_size = aconf.unit_size_proportion_avM
        savepath = Path(config.basepath) / config.experiment_id / 'models'
        return cls(agents, env, preprocessor, input_shape, action_space,
                   aconf.discount, aconf.nstep_return, aconf.replay_size,
                   aconf.replay_min_size, aconf.prioritized_replay,
                   aconf.per_alpha, aconf.per_beta, aconf.per_beta_steps,
                   aconf.noisy_net, aconf.noisy_net_sigma, aconf.eps,
                   aconf.eps_decay, aconf.eps_min, aconf.batch_size,
                   config.test_steps, unit_size, savepath, aconf.double_dqn,
                   aconf.tau_soft_update, config.model_config.model_class,
                   config.model_config, config.optim_config.lr, aconf.nTau1,
                   aconf.nTau2, aconf.k_huber)

    @property
    def env(self):
        return self._env

    def to(self, device):
        """
        Sets current device for pytorch entities
        and sends them to it
        """
        self.device = torch.device(device)
        self.model_b.to(self.device)
        self.model_t.to(self.device)
        for agent in self.agents.values():
            agent.to(device)
        return self

    def idx_to_agent(self, idx: int):
        """
        Returns the agent corresponding to the agent idx.
        Idx 0 is reserved for the null agent
        """
        if idx == 0:
            return None
        return self.agents[self.agent_idxs[idx - 1]]

    @torch.no_grad()
    def get_action(self, state, target=False, device=None):
        """
        External interface - for inference and env interaction
        takes in numpy arrays and returns greedy actions
        """
        qvals = self.get_qvals(state, target=target, device=device)
        actions = qvals.max(-1)[1].squeeze(0)  # (self.n_assets, )
        self.state_cache = state
        return actions

    def action_to_transaction(
            self, actions: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        takes int actions as indexes to agents
        Returns chosen agents' .action_to_transaction method
        """
        assert len(actions.shape) == 1, \
            "action routing meant to be for single batches"
        # multi agent
        # agents = [self.action_to_agent[actions[0][i]]
        #           for i in range(self.n_assets)]
        # single agent
        agent = self.idx_to_agent(actions[0].item())
        with torch.no_grad():
            if agent is not None:
                actions = agent.get_action(self.state_cache, target=True)
                transactions = agent.action_to_transaction(actions)
            else:
                transactions = np.zeros(self.n_assets)
        self.state_cache = None
        return transactions

    # @torch.no_grad()
    # def test_episode(self, test_steps=None, reset=True, target=True) -> dict:
    #     test_steps = test_steps or self.test_steps
    #     if reset:
    #         self.reset_state()
    #     self._preprocessor.initialize_history(
    #         self.env)  # probably already initialized
    #     state = self._preprocessor.current_data()
    #     tst_metrics = []
    #     i = 0
    #     while i <= test_steps:
    #         _tst_metrics = {}
    #         qvals = self.get_qvals(state, target=target)[0].cpu().numpy()
    #         action = self.get_action(state, target=target).cpu().numpy()
    #         transaction = self.action_to_transaction(state, action)
    #         state, reward, done, info = self._env.step(transaction)
    #         self._preprocessor.stream_state(state)
    #         state = self._preprocessor.current_data()
    #         _tst_metrics['qvals'] = qvals
    #         _tst_metrics['reward'] = reward
    #         _tst_metrics['transaction'] = info.brokerResponse.transactionUnits
    #         _tst_metrics[
    #             'transaction_cost'] = info.brokerResponse.transactionCost
    #         # _tst_metrics['info'] = info
    #         tst_metrics.append({**_tst_metrics, **get_env_info(self._env)})
    #         if done:
    #             break
    #         i += 1
    #     return list_2_dict(tst_metrics)
