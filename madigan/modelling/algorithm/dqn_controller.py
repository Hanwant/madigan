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

from .dqn import DQN
from .base import Agent
from . import make_agent
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


class DQNController(DQN):
    """
    Heirarchical DQN.
    The actions of this agent route to a set of pretrained agents.
    The resulting action and transactions are sent to the env.
    Hence it acts a 'controller', enabling a kind of
    meta/heirarchical learning.
    Gradients do not flow to the controlled agents.
    The only difference from normal DQN is the action_to_transaction()
    method which routes to a contained agent, otherwise the interface and
    implementation is exactly the same as DQN.

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

    Might be easiest with Recurrent DQN where actions are to be fed as inputs
    anyway.

    """

    def __init__(self, env, preprocessor, agents: OrderedDict,
                 input_shape: tuple, action_space: ActionSpace,
                 discount: float, nstep_return: int, replay_size: int,
                 replay_min_size: int, noisy_net: bool, noisy_net_sigma: float,
                 eps: float, eps_decay: float, eps_min: float, batch_size: int,
                 test_steps: int, unit_size: float, savepath: Union[Path, str],
                 double_dqn: bool, tau_soft_update: float, model_class: str,
                 model_config: Union[dict, Config], lr: float):
        super().__init__(env, preprocessor, input_shape, action_space,
                         discount, nstep_return, replay_size, replay_min_size,
                         noisy_net, eps, eps_decay, eps_min, batch_size,
                         test_steps, unit_size, savepath)

        assert all(agents[i].action_space == agents[i - 1].action_space
                   for i in range(1, len(
                       agents))), "agents must have same action space"

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agents = agents
        self.agent_idxs = list(self.agents)  # ordered list of agent keys
        self._action_space = action_space
        self.double_dqn = double_dqn
        self.discount = discount
        # safeguard to get 0.001 instead of 0.99
        self.tau_soft_update = min(tau_soft_update, 1 - tau_soft_update)

        self.model_class = get_model_class(type(self).__name__, model_class)
        output_shape = (action_space.n, action_space.action_atoms)
        model_config['noisy_net'] = noisy_net
        model_config['noisy_net_sigma'] = noisy_net_sigma
        self.model_b = self.model_class(input_shape, output_shape,
                                        **model_config)
        self.model_t = self.model_class(input_shape, output_shape,
                                        **model_config)
        self.opt = torch.optim.Adam(self.model_b.parameters(), lr=lr)

        if not self.savepath.is_dir():
            self.savepath.mkdir(parents=True)
        if (self.savepath / 'main.pth').is_file():
            self.load_state()
        else:
            self.model_t.load_state_dict(self.model_b.state_dict())

    @classmethod
    def from_config(cls, config):
        """
        Should be the main entry point to construct agent
        """
        env = make_env(config)
        # ASSUMING AGENTS HAVE SAME PREPROCESSOR FOR NOW
        # NEEDS TO BE SEPARATED FOR EACH AGENT LATER AS
        # DIFFERENT AGENTS MAY HAVE DIFFERENT SPECIALIZATION (I.E LONG RANGE)
        preprocessor = make_preprocessor(config)
        input_shape = preprocessor.feature_output_shape
        agents = OrderedDict({
            name: make_agent(Config.from_path(conf))
            for name, conf in config.agent_configs.items()
        })
        atoms = len(agents) + 1  # +1 for null agent
        action_space = DiscreteRangeSpace((0, atoms), config.n_assets)
        aconf = config.agent_config
        unit_size = aconf.unit_size_proportion_avM
        savepath = Path(config.basepath) / config.experiment_id / 'models'
        return cls(env, preprocessor, agents, input_shape, action_space,
                   aconf.discount, aconf.nstep_return, aconf.replay_size,
                   aconf.replay_min_size, aconf.noisy_net,
                   aconf.noisy_net_sigma, aconf.eps, aconf.eps_decay,
                   aconf.eps_min, aconf.batch_size, config.test_steps,
                   unit_size, savepath, aconf.double_dqn,
                   aconf.tau_soft_update, config.model_config.model_class,
                   config.model_config, config.optim_config.lr)

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

    def action_to_transaction(
            self, state, actions: Union[torch.Tensor,
                                        np.ndarray]) -> np.ndarray:
        """
        takes int actions as indexes to agents
        Returns chosen agents' .action_to_transaction method
        """
        assert actions.shape[0] == 1, \
            "action routing meant to be for single batches"
        # multi agent
        # agents = [self.action_to_agent[actions[0][i]]
        #           for i in range(self.n_assets)]
        # single agent
        agent = self.idx_to_agent(actions[0][0].item())
        if agent is not None:
            actions = agent.get_action(state)
            transactions = agent.action_to_transaction(actions)
        else:
            transactions = np.zeros(self.n_assets)
        return transactions

    def train_step(self, sarsd: SARSD = None):
        self.model_b.sample_noise()
        self.model_t.sample_noise()
        sarsd = self.buffer.sample(self.batch_size) if sarsd is None else sarsd
        state, action, reward, next_state, done = self.prep_sarsd_tensors(
            sarsd)

        action_mask = F.one_hot(action, self.action_atoms).to(self.device)
        qvals = self.model_b(state)
        Qt = (qvals * action_mask).sum(-1)
        Gt = self.calculate_Gt_target(next_state, reward, done)
        assert Qt.shape == Gt.shape

        loss = self.loss_fn(Qt, Gt)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model_b.parameters(),
                                 max_norm=1.,
                                 norm_type=2)
        self.opt.step()

        td_error = (Gt - Qt).abs().mean().detach().item()
        self.update_target()
        return {
            'loss': loss.detach().item(),
            'td_error': td_error,
            'Qt': Qt.detach().mean().item(),
            'Gt': Gt.detach().mean().item()
        }

    def save_state(self, branch="main"):
        # self.save_checkpoint("main")
        state = {
            'state_dict_b': self.model_b.state_dict(),
            'state_dict_t': self.model_t.state_dict(),
            'training_steps': self.training_steps,
            'env_steps': self.env_steps,
            'eps': self.eps
        }
        torch.save(state, self.savepath / f'{branch}.pth')

    def load_state(self, branch="main"):
        state = torch.load(self.savepath / f'{branch}.pth')
        self.model_b.load_state_dict(state['state_dict_b'])
        self.model_t.load_state_dict(state['state_dict_t'])
        self.training_steps = state['training_steps']
        self.env_steps = state['env_steps']
        self.eps = state['eps']

    def _delete_models(self):
        # if self.overwrite_exp:
        saved_models = list(self.savepath.iterdir())
        if len(saved_models):
            for model in saved_models:
                os.remove(model)
