import os
from typing import Union, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .offpolicy_ac import OffPolicyActorCritic
from ..utils import get_model_class
from ...utils import ActionSpace, ContinuousActionSpace
from ...utils.config import Config
from ...utils.data import State, SARSD
from ...utils.preprocessor import make_preprocessor
from ...environments import make_env


class DDPG(OffPolicyActorCritic):
    """
    Implements a DDPG Actor/Critic agent for continuous actions.

    The Agent instance can be called directly to get an action
    based on a state:
        action = agent(state)
    or:
        action = agent.get_action(state)
    use agent.step(n) to step through n environment interactions
    The method for training a single batch is self.train_step(sarsd) where
    sarsd is a class with ndarray members (I.e of shape (bs, time, feats))
    """
    def __init__(self, env, preprocessor, input_shape: tuple,
                 action_space: ActionSpace, discount: float, nstep_return: int,
                 reduce_rewards: bool, reward_shaper_config, replay_size: int,
                 replay_min_size: int, prioritized_replay: bool,
                 per_alpha: float, per_beta: float, per_beta_steps: int,
                 batch_size: int, test_steps: int, savepath: Path,
                 expl_noise_sd: float, double_dqn: bool,
                 tau_soft_update: float, model_class_critic: str,
                 model_class_actor: str, lr_critic: float, lr_actor: float,
                 model_config: Union[dict, Config],
                 proximal_portfolio_penalty: float):
        super().__init__(env, preprocessor, input_shape, action_space,
                         discount, nstep_return, reduce_rewards,
                         reward_shaper_config, replay_size, replay_min_size,
                         prioritized_replay, per_alpha, per_beta,
                         per_beta_steps, batch_size, test_steps, savepath)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._action_space.transform = self.action_to_transaction
        self.expl_noise_sd = expl_noise_sd
        self.double_dqn = double_dqn
        self.discount = discount
        self.tau_soft_update = tau_soft_update
        # output_shape = action_space.output_shape
        self.critic_model_class = get_model_class(
            type(self).__name__, model_class_critic)
        self.actor_model_class = get_model_class(
            type(self).__name__, model_class_actor)

        output_shape = self._action_space.shape
        self.critic_b = self.critic_model_class(input_shape, output_shape,
                                                **model_config)
        self.critic_t = self.critic_model_class(input_shape, output_shape,
                                                **model_config)
        self.actor_b = self.actor_model_class(input_shape, output_shape,
                                              **model_config)
        self.actor_t = self.actor_model_class(input_shape, output_shape,
                                              **model_config)
        self.opt_critic = torch.optim.Adam(self.critic_b.parameters(),
                                           lr=lr_critic)
        self.opt_actor = torch.optim.Adam(self.actor_b.parameters(),
                                          lr=lr_actor)

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
        preprocessor = make_preprocessor(config, env.nAssets)
        input_shape = preprocessor.feature_output_shape
        # add an extra asset for cash holdings
        # used in representation of port weights
        # I.e returned by env.ledgerNormedFull
        action_space = ContinuousActionSpace(-1., 1., env.nAssets + 1, 1)
        aconf = config.agent_config
        savepath = Path(config.basepath) / config.experiment_id / 'models'
        return cls(
            env, preprocessor, input_shape, action_space, aconf.discount,
            aconf.nstep_return, aconf.reduce_rewards,
            config.reward_shaper_config, aconf.replay_size,
            aconf.replay_min_size, aconf.prioritized_replay, aconf.per_alpha,
            aconf.per_beta, aconf.per_beta_steps, aconf.batch_size,
            config.test_steps, savepath, aconf.expl_noise_sd, aconf.double_dqn,
            aconf.tau_soft_update, config.model_config.critic_model_class,
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
        for net in (self.critic_b, self.critic_t, self.actor_b, self.actor_t):
            net.to(self.device)
        return self

    def train_mode(self):
        self.critic_b.train()
        self.critic_t.train()
        self.actor_b.train()
        self.actor_t.train()

    def test_mode(self):
        self.critic_b.eval()
        self.critic_t.eval()
        self.actor_b.eval()
        self.actor_t.eval()

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
                  action: torch.Tensor = None,
                  target=False,
                  device=None):
        """
        External interface - for inference and env interaction
        Takes in numpy arrays
        and return qvals for actions
        """
        device = device or self.device
        state = self.prep_state_tensors(state, device=device)
        if action is None:
            if target:
                action = self.actor_t(state).squeeze(-1)
            else:
                action = self.actor_b(state).squeeze(-1)
        if target:
            return self.critic_t(state, action)
        return self.critic_b(state, action)

    @torch.no_grad()
    def get_action(self, state, target=False, device=None):
        """
        External interface - for inference and env interaction
        takes in numpy arrays and returns greedy actions
        """
        device = device or self.device
        state = self.prep_state_tensors(state, device=device)
        if target:
            return self.actor_t(state).squeeze(-1)
        return self.actor_b(state).squeeze(-1)

    def explore(self,
                state,
                noise_sd: float = None) -> Tuple[np.ndarray, np.ndarray]:
        noise_sd = noise_sd or self.expl_noise_sd
        action = self.get_action(state, target=False).cpu()
        action += noise_sd * torch.randn_like(action)
        return action.numpy()[0], self.action_to_transaction(action)[0]

    def action_to_transaction(self, actions: torch.Tensor) -> np.ndarray:
        """
        Takes output from net and converts to transaction units
        Given current ledgerNormedFull - portfolio weights (incl cash)
        and current prices
        """
        assert actions.shape[1:] == (self.n_assets, )
        # cash_pct = self.env.cash / self.env.equity
        # current_port = np.concatenate(([cash_pct], self.env.ledgerNormed))
        # actions = actions.cpu().numpy()
        if actions.sum(-1) == 0:  # prevent div by 0.
            desired_port = actions
        else:
            desired_port = actions / actions.sum(axis=-1)
        # max_amount = self.env.availableMargin * 0.25
        current_port = self.env.ledgerNormedFull
        assert abs(current_port.sum() - 1.) < 1e-8
        # assert abs(desired_port.sum() - 1.) < 1e-8
        # if not abs(desired_port.sum() - 1.) < 1e-5:
        #     import ipdb
        #     ipdb.set_trace()
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

    def critic_loss_fn(self, Q_t, G_t, weights):
        return self.critic_loss_fn_mse(Q_t, G_t, weights)

    def critic_loss_fn_huber(self, Q_t, G_t, weights: torch.Tensor = None):
        loss = nn.functional.smooth_l1_loss(Q_t, G_t, reduce=False)
        assert loss.shape == (Q_t.shape[0], )
        if weights is None:
            return loss.mean()
        return (loss * weights).mean()

    def critic_loss_fn_mse(self, Q_t, G_t, weights: torch.Tensor = None):
        loss = nn.functional.mse_loss(Q_t, G_t, reduce=False)
        assert loss.shape == (Q_t.shape[0], )
        if weights is None:
            return loss.mean()
        return (loss * weights).mean()

    @torch.no_grad()
    def calculate_Gt_target(self, next_state, reward, done):
        """
        Given a next_state State object, calculates the target value
        to be used in td error and loss calculation
        """
        next_action = self.actor_t(next_state).squeeze(-1)
        qvals_next = self.critic_t(next_state, next_action)
        # reward and done have an extended dimension to accommodate for n_assets
        # As actions for different assets are considered in parallel
        Gt = reward[..., None] + (~done[..., None] *
                                  (self.discount**self.nstep_return) *
                                  qvals_next)  # Gt = (bs, n_assets)
        return Gt

    def train_step(self, sarsd: SARSD = None, weights: np.ndarray = None):
        sarsd, weights = self.buffer.sample(
            self.batch_size) if sarsd is None else (sarsd, weights)
        state, action, reward, next_state, done = \
            self.prep_sarsd_tensors(sarsd)
        loss_critic, Qt, Gt, td_error = self.train_step_critic(
            state, action, reward, next_state, done, weights)
        loss_actor = self.train_step_actor(state)
        self.update_critic_target()
        self.update_actor_target()
        return {
            'loss_critic': loss_critic,
            'loss_actor': loss_actor,
            'td_error': td_error.detach().mean().item(),
            'Qt': Qt.detach().mean().item(),
            'Gt': Gt.detach().mean().item()
        }

    def train_step_critic(self, state, action, reward, next_state, done,
                          weights):
        Qt = self.critic_b(state, action).mean(-1)  # mean across assets
        Gt = self.calculate_Gt_target(next_state, reward, done).mean(-1)
        assert Qt.shape == Gt.shape

        td_error = (Gt - Qt).abs().detach()
        if self.prioritized_replay:
            self.buffer.update_priority(td_error.squeeze())
            weights = torch.from_numpy(weights).to(self.device)

        loss_critic = self.critic_loss_fn(Qt, Gt, weights)

        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        return loss_critic.detach().item(), Qt, Gt, td_error

    def train_step_actor(self, state, weights=None):
        action = self.actor_b(state).squeeze(-1)
        loss_actor = -self.critic_t(state, action).mean()
        port_penalty = ((action - state.portfolio)**2).mean()
        # norm_penalty = (1.*action.shape[0] - action.sum()) ** 2
        loss_actor += self.proximal_portfolio_penalty * port_penalty
        # loss_actor += self.norm_penalty * norm_penalty
        self.opt_actor.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor_b.parameters(),
                                 max_norm=1.,
                                 norm_type=2)
        self.opt_actor.step()
        return loss_actor.detach().item()

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
            'state_dict_actor_b': self.actor_b.state_dict(),
            'state_dict_actor_t': self.actor_t.state_dict(),
            'training_steps': self.training_steps,
            'env_steps': self.env_steps
        }
        torch.save(state, self.savepath / f'{branch}.pth')

    def load_state(self, branch=None):
        branch = branch or "main"
        state = torch.load(self.savepath / f'{branch}.pth')
        self.critic_b.load_state_dict(state['state_dict_critic_b'])
        self.critic_t.load_state_dict(state['state_dict_critic_t'])
        self.actor_b.load_state_dict(state['state_dict_actor_b'])
        self.actor_t.load_state_dict(state['state_dict_actor_t'])
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

    def update_actor_target(self):
        """
        Soft Update
        """
        for behaviour, target in zip(self.actor_b.parameters(),
                                     self.actor_t.parameters()):
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


class DDPGDiscretized(DDPG):
    @classmethod
    def from_config(cls, config):
        inst = super().from_config(config)
        aconf = config.agent_config
        inst.transaction_thresh = aconf.transaction_thresh
        return inst

    def action_to_transaction(self, actions: torch.Tensor) -> np.ndarray:
        """
        Takes output from net and converts to transaction units
        Given current ledgerNormedFull - portfolio weights (incl cash)
        and current prices
        """
        current_port = self.env.ledgerNormedFull
        # actions = actions.cpu().numpy()
        desired_port = actions / actions.sum(axis=-1)
        # max_amount = self.env.availableMargin * 0.25
        amount_weights = (desired_port - current_port)
        # amounts = ((desired_port - current_port) * self.env.equity
        #            ).clip(-max_amount, max_amount)
        amount_weights = np.where(
            amount_weights.abs() < self.transaction_thresh, 0., amount_weights)
        amounts = amount_weights * self.env.equity
        units = amounts[..., 1:] / self.env.currentPrices  # element wise div
        return units
