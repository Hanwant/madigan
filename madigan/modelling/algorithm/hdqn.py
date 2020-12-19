import os
from typing import Union
from pathlib import Path
from random import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dqn import DQN
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


class HDQN:
    """
    Implements a base DQN agent from which extensions can inherit
    The Agent instance can be called directly to get an action based on a state:
        action = dqn(state)
    or:
        action = dqn.get_action(state)
    use dqn.step(n) to step through n environment interactions
    The method for training a single batch is self.train_step(sarsd) where sarsd is a class with ndarray members (I.e of shape (bs, time, feats))
    """
    def __init__(self, env, agents: list):

        assert all(agents[i].action_space == agents[i-1].action_space
                   for i in range(1, len(agents))), "agents must have same action space"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._action_space = agents[0].action_space
        output_shape = (self._action_space.n,
                        self._action_space.action_atoms)
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
        agents = [make_agent(Config.from_path(conf))
                  for conf in config.agent_configs]
        savepath = Path(config.basepath) / config.experiment_id / 'models'
        return cls(env, agents, savepath)

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
        return self

    def train_mode(self):
        """
        Called at start of training.
        Necessary when using modules like nn.BatchNorm and nn.Dropout
        """
        self.model_b.train()
        self.model_t.train()

    def test_mode(self):
        """
        Called before testing and performing inference.
        Necessary when using modules like nn.BatchNorm and nn.Dropout
        """
        self.model_b.eval()
        self.model_t.eval()

    @property
    def action_space(self) -> np.ndarray:
        """
        Action space object which can be sampled from
        outputs transaction units
        """
        # units = self.unit_size * self.env.availableMargin \
        #     / self.env.currentPrices
        # self._action_space.action_multiplier = units
        return self._action_space

    def action_to_transaction(
            self, actions: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Takes output from net and converts to transaction units
        """
        units = self.unit_size * self._env.availableMargin \
            / self._env.currentPrices
        actions_centered = (actions - (self.action_atoms // 2))
        if isinstance(actions_centered, torch.Tensor):
            actions_centered = actions_centered.cpu().numpy()
        transactions = actions_centered * units
        # Reverse position if action is '0' and position exists
        for i, act in enumerate(actions):
            if act == 0:
                current_holding = self._env.ledger[i]
                if current_holding != 0:
                    transactions[i] = -current_holding
                else:
                    transactions[i] = 0.
        return transactions

    @torch.no_grad()
    def get_qvals(self, state, target=False, device=None):
        """
        External interface - for inference and env interaction
        Takes in numpy arrays
        and return qvals for actions
        """
        device = device or self.device
        state = self.prep_state_tensors(state, device=device)
        if target:
            return self.model_t(state)
        return self.model_b(state)

    @torch.no_grad()
    def get_action(self, state, target=False, device=None):
        """
        External interface - for inference and env interaction
        takes in numpy arrays and returns greedy actions
        """
        qvals = self.get_qvals(state, target=target, device=device)
        actions = qvals.max(-1)[1][:, 0]  # get rid of last dim
        return actions

    def __call__(self,
                 state: State,
                 target: bool = False,
                 raw_qvals: bool = False,
                 max_qvals: bool = False):
        return self.get_action(state, target=target)

    def prep_state_tensors(self, state, batch=False, device=None):
        if not batch:
            price = torch.as_tensor(state.price[None, ...],
                                    dtype=torch.float32).to(self.device)
            port = torch.as_tensor(state.portfolio[None, -1],
                                   dtype=torch.float32).to(self.device)
        else:
            price = torch.as_tensor(state.price,
                                    dtype=torch.float32).to(self.device)
            port = torch.as_tensor(state.portfolio[:, -1],
                                   dtype=torch.float32).to(self.device)
        return State(price, abs_port_norm(port), state.timestamp)

    def prep_sarsd_tensors(self, sarsd, device=None):
        state = self.prep_state_tensors(sarsd.state, batch=True)
        action = torch.as_tensor(sarsd.action,
                                 dtype=torch.long,
                                 device=self.device)  # [..., 0]
        reward = torch.as_tensor(sarsd.reward,
                                 dtype=torch.float32,
                                 device=self.device)
        next_state = self.prep_state_tensors(sarsd.next_state, batch=True)
        done = torch.as_tensor(sarsd.done,
                               dtype=torch.bool,
                               device=self.device)
        return state, action, reward, next_state, done

    def loss_fn(self, Q_t, G_t):
        return F.smooth_l1_loss(Q_t, G_t)

    @torch.no_grad()
    def calculate_Gt_target(self, next_state, reward, done):
        """
        Given a next_state State object, calculates the target value
        to be used in td error and loss calculation
        """
        if self.double_dqn:
            behaviour_actions = self.model_b(next_state).max(-1)[1]
            one_hot = F.one_hot(behaviour_actions,
                                self.action_atoms).to(self.device)
            greedy_qvals_next = (self.model_t(next_state) * one_hot).sum(
                -1)  # (bs, n_assets, 1)
        else:
            greedy_qvals_next = self.model_t(next_state).max(-1)[
                0]  # (bs, n_assets, 1)
        # reward and done have an extended dimension to accommodate for n_assets
        # As actions for different assets are considered in parallel
        Gt = reward[..., None] + (~done[..., None] *
                                  (self.discount**self.nstep_return) *
                                  greedy_qvals_next)  # Gt = (bs, n_assets)
        return Gt

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

    def update_target_hard(self):
        """ Hard update, copies weights """
        self.model_t.load_state_dict(self.model_b.state_dict())

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
        # else:
        #     raise NotImplementedError("Attempting to delete models when config.overwrite_exp is not set to true")

    def update_target(self):
        """
        Soft Update
        """
        for behaviour, target in zip(self.model_b.parameters(),
                                     self.model_t.parameters()):
            target.data.copy_(self.tau_soft_update * behaviour.data + \
                              (1.-self.tau_soft_update)*target.data)

    def filter_transactions(self, transactions, portfolio):
        """
        Prevents doubling up on positions
        """
        for i, action in enumerate(transactions):
            if portfolio[i] == 0.:
                pass
            elif np.sign(portfolio[i]) == np.sign(action):
                transactions[i] = 0.
        return transactions


class DQNCURL(DQN):
    """
    CURL: Contrastive Unsupervised Representation Learning
    This agent mainly just wraps the appropriate CURL-enabled nn.Module which
    contains the actual functionality for performing CURL.
    Keeping the main logic in the contained nn model maintains code resuability
    at the higher abstraction of the agent and lets the model take care of
    internal housekeeping (I.e defining and updating key encoder).
    """
    def train_step(self, sarsd: SARSD = None):
        """
        wraps train_step() of DQN to include the curl objective
        """
        sarsd = sarsd or self.buffer.sample(self.batch_size)
        state = self.prep_state_tensors(sarsd.state, batch=True)
        # contrastive unsupervised objective
        loss_curl = self.model_b.train_contrastive_objective(state)
        # do normal rl training objective and add 'loss_curl' to output dict
        return {'loss_curl': loss_curl, **super().train_step(sarsd)}


# For temporary backwards-comp
DQNReverser = DQN
