import os
from typing import Union
from pathlib import Path
import math

import numpy as np
import torch
# import torch.functional
import torch.nn as nn

from .offpolicy_ac import OffPolicyActorCritic
from .utils import abs_port_norm
from ..utils import get_model_class
from ...utils import ActionSpace, DiscreteRangeSpace, list_2_dict
from ...utils.config import Config
from ...utils.data import State
from ...utils.preprocessor import make_preprocessor
from ...environments import make_env, get_env_info


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
                 test_steps: int, savepath: Union[Path, str],
                 learn_entropy_temp: bool, entropy_temp: float,
                 target_entropy_ratio: float,
                 double_dqn: bool, tau_soft_update: float,
                 model_class_critic: str, model_class_actor: str,
                 lr_critic: float, lr_actor: float,
                 model_config: Union[dict, Config], unit_size: float):
        super().__init__(env, preprocessor, input_shape, action_space,
                         discount, nstep_return, replay_size, replay_min_size,
                         batch_size, test_steps, savepath)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._action_space.transform = self.action_to_transaction
        self.double_dqn = double_dqn
        self.discount = discount
        self.tau_soft_update = tau_soft_update
        self.unit_size = unit_size
        # output_shape = action_space.output_shape
        self.critic_model_class = get_model_class(
            type(self).__name__, model_class_critic)
        self.actor_model_class = get_model_class(
            type(self).__name__, model_class_actor)

        output_shape = (action_space.n, action_space.action_atoms)
        self.critic_b = self.critic_model_class(input_shape, output_shape,
                                                **model_config)
        self.critic_t = self.critic_model_class(input_shape, output_shape,
                                                **model_config)
        for param in self.critic_t.modules():
            param.requires_grad = False

        self.actor = self.actor_model_class(input_shape, output_shape,
                                            **model_config)
        self.opt_critic1 = torch.optim.Adam(self.critic_b.Q1.parameters(),
                                            lr=lr_critic)
        self.opt_critic2 = torch.optim.Adam(self.critic_b.Q2.parameters(),
                                            lr=lr_critic)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.learn_entropy_temp = learn_entropy_temp
        # 1/|A| == max entropy
        self.target_entropy_ratio = target_entropy_ratio
        if learn_entropy_temp:
            self.target_entropy = \
                -math.log(1. / self.action_space.action_atoms) * \
                target_entropy_ratio

            self.log_temp = torch.zeros(1, requires_grad=True,
                                        device=self.device)
            self.temp = self._temp
            self.opt_entropy_temp = torch.optim.Adam([self.log_temp],
                                                     lr=lr_actor)
        else:
            self.temp = entropy_temp

        if not self.savepath.is_dir():
            self.savepath.mkdir(parents=True)
        if (self.savepath / 'main.pth').is_file():
            self.load_state()
        else:
            self.critic_t.load_state_dict(self.critic_b.state_dict())

    @property
    def _temp(self):
        """
        If self.learn_entropy_temp, then this is used for self.temp
        """
        return self.log_temp.exp()

    @classmethod
    def from_config(cls, config):
        env = make_env(config)
        preprocessor = make_preprocessor(config)
        input_shape = preprocessor.feature_output_shape
        # add an extra asset for cash holdings
        # used in repr of port weights returned by env.ledgerNormedFull
        atoms = config.discrete_action_atoms + 1
        action_space = DiscreteRangeSpace((0, atoms), config.n_assets)
        aconf = config.agent_config
        unit_size = aconf.unit_size_proportion_avM
        savepath = Path(config.basepath) / config.experiment_id / 'models'
        return cls(env, preprocessor, input_shape, action_space,
                   aconf.discount, aconf.nstep_return, aconf.replay_size,
                   aconf.replay_min_size, aconf.batch_size, config.test_steps,
                   savepath, aconf.learn_entropy_temp, aconf.entropy_temp,
                   aconf.target_entropy_ratio, aconf.double_dqn,
                   aconf.tau_soft_update,
                   config.model_config.critic_model_class,
                   config.model_config.actor_model_class,
                   config.optim_config.lr_critic, config.optim_config.lr_actor,
                   config.model_config, unit_size)

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
    def get_qvals(self,
                  state: State,
                  actions: torch.Tensor = None,
                  target: bool = False):
        """
        External interface - for inference and env interaction
        Takes in numpy arrays
        and return qvals for actions
        """
        state = self.prep_state_tensors(state)
        if target:
            q1, q2 = self.critic_t(state)
        else:
            q1, q2 = self.critic_b(state)
        if actions is not None:
            if len(actions.shape) == 2:
                actions = actions[..., None]
            actions = torch.LongTensor(actions).to(state.price.device)
            q1 = q1.gather(-1, actions)[..., 0]
            q2 = q2.gather(-1, actions)[..., 0]
        return q1, q2

    @torch.no_grad()
    def get_action(self, state: State, target: bool = False):
        """
        External interface - for inference and env interaction
        takes in numpy arrays and returns greedy actions
        """
        state = self.prep_state_tensors(state)
        return self.actor.get_action(state).squeeze(-1).cpu().numpy()

    def explore(self, state: State):
        """
        Used when interacting with env and collecting experiences
        """
        state = self.prep_state_tensors(state)
        action, action_p, log_action_p = self.actor.sample(state)
        return action

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

    def get_transactions(self, state, target=False, device=None):
        actions = self.get_action(state, target, device)
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
        return State(price, abs_port_norm(port), state.timestamp)

    def prep_sarsd_tensors(self, sarsd, device=None):
        state = self.prep_state_tensors(sarsd.state, batch=True)
        action = torch.as_tensor(sarsd.action,
                                 dtype=torch.long,
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
        qvals_next = (action_p * (torch.min(qvals_next1, qvals_next2) -
                                  self.temp * log_action_p)
                      ).sum(dim=-1, keepdim=False)

        # reward and done have an extended dimension to accommodate for n_assets
        # As actions for different assets are considered in parallel
        Gt = reward[..., None] + (~done[..., None] *
                                  (self.discount**self.nstep_return) *
                                  qvals_next)  # Gt = (bs, n_assets)
        return Gt

    def train_step_critic(self, state, action, reward, next_state, done):
        Qt1, Qt2 = self.critic_b(state)
        Qt1 = Qt1.gather(-1, action)[..., 0]
        Qt2 = Qt2.gather(-1, action)[..., 0]
        Gt = self.calculate_Gt_target(next_state, reward, done)

        loss_critic1 = nn.functional.mse_loss(Qt1, Gt)
        loss_critic2 = nn.functional.mse_loss(Qt2, Gt)

        self.opt_critic1.zero_grad()
        self.opt_critic2.zero_grad()
        loss_critic1.backward()
        loss_critic2.backward()
        self.opt_critic1.step()
        self.opt_critic2.step()

        return (loss_critic1.detach().item(), loss_critic2.detach().item(),
                Qt1.detach(), Qt2.detach(), Gt)

    def train_step_actor(self, state):
        _, action_p, log_action_p = self.actor.sample(state)
        with torch.no_grad():
            Qt1, Qt2 = self.critic_b(state)

        entropy = -(action_p * log_action_p).sum(dim=2, keepdim=True)
        Qt = (torch.min(Qt1, Qt2) * action_p).sum(dim=2, keepdim=True)

        loss_actor = -(Qt + self.temp * entropy).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()
        return loss_actor.detach().item(), entropy.detach()

    def train_step_entropy(self, entropy):
        assert not entropy.requires_grad
        loss_entropy = -(self.log_temp *
                         (self.target_entropy - entropy)).mean()
        self.opt_entropy_temp.zero_grad()
        loss_entropy.backward()
        self.opt_entropy_temp.step()
        return loss_entropy.detach().item()

    def train_step(self, sarsd=None):
        sarsd = sarsd or self.buffer.sample(self.batch_size)
        state, action, reward, next_state, done = \
            self.prep_sarsd_tensors(sarsd)
        loss_critic1, loss_critic2, Qt1, Qt2, Gt = \
            self.train_step_critic(state, action, reward, next_state, done)
        loss_actor, entropy = self.train_step_actor(state)
        if self.learn_entropy_temp:
            loss_entropy = self.train_step_entropy(entropy)
            temp = self.temp.item()
        else:
            loss_entropy = 0.
            temp = self.temp
        Qt = torch.min(Qt1, Qt2)
        td_error = (Qt - Gt).abs()
        self.update_critic_target()
        return {
            'loss_critic1': loss_critic1,
            'loss_critic2': loss_critic2,
            'loss_actor': loss_actor,
            'loss_entropy': loss_entropy,
            'entropy': entropy.mean().item(),
            'td_error': td_error.detach().mean().item(),
            'Qt1': Qt1.detach().mean().item(),
            'Qt2': Qt2.detach().mean().item(),
            'Gt': Gt.detach().mean().item(),
            'entropy_temp': temp
        }

    def update_targets_hard(self):
        """ Hard update, copies weights """
        self.critic_t.load_state_dict(self.critic_b.state_dict())

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
        May be useful for risk averse testing.
        """
        for i, action in enumerate(transactions):
            if portfolio[i] == 0.:
                pass
            elif np.sign(portfolio[i]) == np.sign(action):
                transactions[i] = 0.
        return transactions

    def get_action_distribution(self, state: State):
        state = self.prep_state_tensors(state)
        action_sampled, action_p, log_action_p = self.actor.sample(state)
        return action_sampled, action_p, log_action_p

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
            action_greedy = self.get_action(state, target=target)
            qvals1, qvals2 = (qval[0].cpu().numpy() for qval in self.get_qvals(
                state, actions=action_greedy, target=target))
            transaction = self.action_to_transaction(action_greedy)
            action_sample, action_p, log_action_p = \
                self.get_action_distribution(state)
            state, reward, done, info = self._env.step(transaction)
            self._preprocessor.stream_state(state)
            state = self._preprocessor.current_data()
            _tst_metrics['qvals1'] = qvals1
            _tst_metrics['qvals2'] = qvals2
            _tst_metrics['action_probs'] = action_p[0].cpu().numpy()
            _tst_metrics['reward'] = reward
            _tst_metrics['action'] = action_greedy
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
