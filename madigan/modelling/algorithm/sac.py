import os
from typing import Union
from pathlib import Path
import math

import numpy as np
import torch
# import torch.functional
import torch.nn as nn

from .offpolicy_ac import OffPolicyActorCritic
from ..utils import get_model_class
from ...utils import ActionSpace, ContinuousActionSpace
from ...utils.config import Config
from ...utils.data import State
from ...utils.preprocessor import make_preprocessor
from ...environments import make_env


class SACDiscrete(OffPolicyActorCritic):
    """
    Implements a Discrete Soft Actor Critic agent for continuous actions.
    The crux of the soft actor critic is that along with expected reward,
    it aims to maximize entropy of its actions, yielding favourable exploration
    and generalization behaviour.

    The Agent instance can be called directly to get an action
    based on a state:
        action = agent(state)
    or:
        action = agent.get_action(state)
    use agent.step(n) to step through n environment interactions
    The method for training a single batch is:
        self.train_step(sarsd)
    where sarsd is of type SARSD or:
        self.train_step()
    where the sarsd is sampled from the internal buffer.

    Implementatation follows https://github.com/ku2482/sac-discrete.pytorch
    """
    def __init__(self, env, preprocessor, input_shape: tuple,
                 action_space: ActionSpace, discount: float, nstep_return: int,
                 replay_size: int, replay_min_size: int, batch_size: int,
                 target_entropy_ratio: float, test_steps: int,
                 savepath: Union[Path, str], double_dqn: bool,
                 tau_soft_update: float, model_class_critic: str,
                 model_class_actor: str, lr_critic: float, lr_actor: float,
                 model_config: Union[dict, Config],
                 proximal_portfolio_penalty: float):
        super().__init__(env, preprocessor, input_shape, action_space,
                         discount, nstep_return, replay_size, replay_min_size,
                         batch_size, expl_noise_sd, test_steps, savepath)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._action_space.transform = self.action_to_transaction
        self.double_dqn = double_dqn
        self.discount = discount
        self.tau_soft_update = tau_soft_update
        # output_shape = action_space.output_shape
        self.critic_model_class = get_model_class(
            type(self).__name__, model_class_critic)
        self.actor_model_class = get_model_class(
            type(self).__name__, model_class_actor)

        # 1/|A| == max entropy
        self.target_entropy = \
            -math.log(1. / self.action_space.shape[1]) * target_entropy_ratio
        self.log_temp = torch.zeros(1, requires_grad=True, device=self.device)
        self.temp = self.log_temp.exp()

        self.critic_b = self.critic_model_class(input_shape, self.n_assets, 1,
                                                **model_config)
        self.critic_t = self.critic_model_class(input_shape, self.n_assets, 1,
                                                **model_config)
        for param in self.critic_t.modules():
            param.requires_grad = False

        self.actor = self.actor_model_class(input_shape, self.n_assets, 1,
                                            **model_config)
        self.opt_critic1 = torch.optim.Adam(self.critic_b.Q1.parameters(),
                                            lr=lr_critic)
        self.opt_critic2 = torch.optim.Adam(self.critic_b.Q2.parameters(),
                                            lr=lr_critic)
        self.opt_actor = torch.optim.Adam(self.actor_b.parameters(),
                                          lr=lr_actor)
        self.opt_temp = torch.optim.Adam([self.log_alpha], lr=lr_actor)

        if not self.savepath.is_dir():
            self.savepath.mkdir(parents=True)
        if (self.savepath / 'main.pth').is_file():
            self.load_state()
        else:
            self.critic_t.load_state_dict(self.critic_b.state_dict())
            self.actor_t.load_state_dict(self.actor_b.state_dict())
        self.proximal_portfolio_penalty = proximal_portfolio_penalty
        self.norm_penalty = 1.

    @classmethod
    def from_config(cls, config):
        env = make_env(config)
        preprocessor = make_preprocessor(config)
        input_shape = preprocessor.feature_output_shape
        # add an extra asset for cash holdings
        # used in representation of port weights
        # I.e returned by env.ledgerNormedFull
        action_space = ContinuousActionSpace(-1., 1., config.n_assets + 1, 1)
        aconf = config.agent_config
        savepath = Path(config.basepath) / config.experiment_id / 'models'
        return cls(env, preprocessor, input_shape, action_space,
                   aconf.discount, aconf.nstep_return, aconf.replay_size,
                   aconf.replay_min_size, aconf.batch_size,
                   aconf.target_entropy_ratio, config.test_steps, savepath,
                   aconf.double_dqn, aconf.tau_soft_update,
                   config.model_config.critic_model_class,
                   config.model_config.actor_model_class,
                   config.optim_config.lr_critic, config.optim_config.lr_actor,
                   config.model_config, aconf.proximal_portfolio_penalty)

    @property
    def env(self):
        return self._env

    def to(self, device):
        """
        Sets current device for pytorch entities
        and sends them to it
        """
        self.device = torch.device(device)
        for net in (self.critic_b, self.critic_t, self.actor):
            net.to(self.device)
        return self

    def train_mode(self):
        self.critic_b.train()
        self.critic_t.train()
        self.actor.train()

    def test_mode(self):
        self.critic_b.eval()
        self.critic_t.eval()
        self.actor.eval()

    @property
    def action_space(self) -> np.ndarray:
        """
        Action space object which can be sampled from
        outputs transaction units
        """
        return self._action_space

    @torch.no_grad()
    def get_qvals(self, state: State, target: bool = False):
        """
        External interface - for inference and env interaction
        Takes in numpy arrays
        and return qvals for actions
        """
        state = self.prep_state_tensors(state)
        if target:
            return self.critic_t(state)
        return self.critic_b(state)

    @torch.no_grad()
    def get_action(self, state: State, target: bool = False)
        """
        External interface - for inference and env interaction
        takes in numpy arrays and returns greedy actions
        """
        state = self.prep_state_tensors(state)
        return self.actor.get_action(state).squeeze(-1)

    def explore(self, state: State):
        """
        Used when interacting with env and collecting experiences
        """
        action, action_p, log_action_p = self.actor.sample(state)
        return action


    def action_to_transaction(self, actions: torch.Tensor) -> np.ndarray:
        """
        Takes output from net and converts to transaction units
        Given current ledgerNormedFull - portfolio weights (incl cash)
        and current prices
        """
        assert actions.shape[1:] == (self.n_assets, )
        # cash_pct = self.env.cash / self.env.equity
        # current_port = np.concatenate(([cash_pct], self.env.ledgerNormed))
        current_port = self.env.ledgerNormedFull
        assert abs(current_port.sum() - 1.) < 1e-8
        # actions = actions.cpu().numpy()
        desired_port = actions / actions.sum(axis=-1)
        # max_amount = self.env.availableMargin * 0.25
        amounts = (desired_port - current_port) * self.env.equity
        # amounts = ((desired_port - current_port) * self.env.equity
        #            ).clip(-max_amount, max_amount)
        units = amounts[..., 1:] / self.env.currentPrices  # element wise div
        return units

    def get_transactions(self, state, target=False, device=None):
        actions = self.get_actions(state, target, device)
        transactions = self.actions_to_transactions(actions)
        return transactions

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


#         timestamp = torch.as_tensor(state.timestamp)
        return State(price, port, state.timestamp)

    def prep_sarsd_tensors(self, sarsd, device=None):
        state = self.prep_state_tensors(sarsd.state, batch=True)
        #         action = np.rint(sarsd.action // self.lot_unit_value) + self.action_atoms//2
        # action = self.transactions_to_actions(sarsd.action)
        action = torch.as_tensor(sarsd.action,
                                 dtype=torch.float32,
                                 device=self.device)
        reward = torch.as_tensor(sarsd.reward,
                                 dtype=torch.float32,
                                 device=self.device)
        next_state = self.prep_state_tensors(sarsd.next_state, batch=True)
        done = torch.as_tensor(sarsd.done,
                               dtype=torch.bool,
                               device=self.device)
        return state, action, reward, next_state, done

    def loss_fn(self, Q_t, G_t):
        return nn.functional.smooth_l1_loss(Q_t, G_t)

    @torch.no_grad()
    def calculate_Gt_target(self, next_state, reward, done):
        """
        Given a next_state State object, calculates the target value
        to be used in td error and loss calculation
        """
        action, action_p, log_action_p = \
            self.actor.sample(next_state)
        qvals_next1, qvals_next2 = self.critic_t(next_state)
        qvals_next = (action_p * (torch.min(qvals_next1, qvals_next2)
                                  - self.temp*log_action_p)
                      ).sum(dim=-1, keepdim=True)
        print(action_p.shape, qvals_next.shape,
              torch.min(qvals_next1, qvals_next2).shape,
              torch.min(qvals_next1, qvals_next2, dim=-1).shape)

        # reward and done have an extended dimension to accommodate for n_assets
        # As actions for different assets are considered in parallel
        Gt = reward[..., None] + (~done[..., None] *
                                  (self.discount**self.nstep_return) *
                                  qvals_next)  # Gt = (bs, n_assets)
        return Gt

    def train_step(self, sarsd=None):
        sarsd = sarsd or self.buffer.sample(self.batch_size)
        state, action, reward, next_state, done = \
            self.prep_sarsd_tensors(sarsd)
        loss_critic1, loss_critic2, Qt1, Qt2, Gt = \
            self.train_step_critic(state, action, reward, next_state, done)
        loss_actor, entropy = self.train_step_actor(state, action, reward,
                                                    next_state, done)
        loss_entropy = self.train_step_entropy(entropy)
        td_error = (Qt - Gt).abs()
        self.update_critic_target()
        self.update_actor_target()
        return {
            'loss_critic1': loss_critic1,
            'loss_critic2': loss_critic2,
            'loss_actor': loss_actor,
            'loss_entropy': loss_entropy,
            'entropy': entropy,
            'td_error': td_error.detach().mean().item(),
            'Qt1': Qt1.detach().mean().item(),
            'Qt2': Qt2.detach().mean().item(),
            'Gt': Gt.detach().mean().item()
        }

    def train_step_critic(self, state, action, reward, next_state, done):
        Qt1, Qt2 = self.critic_b(state)
        Gt = self.calculate_Gt_target(next_state, reward, done)
        assert Qt.shape == Gt.shape

        loss_critic1 = self.loss_fn(Qt, Gt)
        loss_critic2 = self.loss_fn(Qt, Gt)

        self.opt_critic1.zero_grad()
        self.opt_critic2.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        self.opt_critic1.step()
        self.opt_critic2.step()

        return (loss_critic1.detach().item(), loss_critic1.detach().item(),
                Qt1, Qt2, Gt)

    def train_step_actor(self, state, action, reward, next_state, done):
        action, action_p, log_action_p = self.actor.sample(state)

        with torch.no_grad():
            Qt1, Qt2 = self.critic_b(states)

        Qt = (torch.min(Qt1, Qt2) * action_p).sum(dim=-1, keepdim=True)

        entropy = -(action_p * log_action_p).sum(dim=-1, keepdim=True)
        loss_actor = -(Qt - self.temp * entropy).mean()
        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()
        return loss_actor.detach().item(), entropy.detach()

    def train_step_entropy(self, entropy):
        assert not entropy.requires_grad
        entropy_loss = -(self.log_temp * (self.target_entropy - entropy)).mean()
        self.opt_entropy.zero_grad()
        entropy_loss.backward()
        self.opt_entropy.step()
        return entropy_loss.detach().item()

    def update_targets_hard(self):
        """ Hard update, copies weights """
        self.critic_t.load_state_dict(self.critic_b.state_dict())
        self.actor_t.load_state_dict(self.actor_b.state_dict())

    def save_state(self, branch=None):
        branch = branch or "main"
        # self.save_checkpoint("main")
        state = {
            'state_dict_critic_b': self.critic_b.state_dict(),
            'state_dict_critic_t': self.critic_t.state_dict(),
            'state_dict_actor': self.actor.state_dict(),
            'training_steps': self.training_steps,
            'env_steps': self.env_steps
        }
        torch.save(state, self.savepath / f'{branch}.pth')

    def load_state(self, branch=None):
        branch = branch or "main"
        state = torch.load(self.savepath / f'{branch}.pth')
        self.critic_b.load_state_dict(state['state_dict_critic_b'])
        self.critic_t.load_state_dict(state['state_dict_critic_t'])
        self.actor.load_state_dict(state['state_dict_actor'])
        self.training_steps = state['training_steps']
        self.env_steps = state['env_steps']

    def _delete_models(self):
        saved_models = list(self.savepath.iterdir())
        if len(saved_models):
            for model in saved_models:
                os.remove(model)

    def update_critic_target(self):
        """
        Soft Update
        """
        for behaviour, target in zip(self.critic_b.parameters(),
                                     self.critic_t.parameters()):
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

