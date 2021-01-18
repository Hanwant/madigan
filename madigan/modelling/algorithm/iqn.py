import os
from typing import Union
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

from .dqn import DQN, DQNMixedActions
from ...environments import make_env, get_env_info
from ..net.conv_net_iqn import ConvNetIQN
from ...utils import default_device, DiscreteActionSpace, DiscreteRangeSpace
from ...utils.preprocessor import make_preprocessor
from ...utils.metrics import list_2_dict
from ...utils.config import Config
from ...utils.data import State, SARSD

std_normal_dist = torch.distributions.Normal(0, 1)
cliiped_uniform_dist = Uniform(1e-7, 1 - 1e-7)


def make_risk_distortion(risk_distortion_type: str,
                         risk_distortion_param: float):
    if risk_distortion_type in ('None', 'none', 'neutral', None):
        return lambda x: x
    if risk_distortion_type in globals():
        return partial(globals()[risk_distortion_type], risk_distortion_param)
    risk_distortion_type += '_distortion'
    if risk_distortion_type in globals():
        return partial(globals()[risk_distortion_type], risk_distortion_param)
    raise NotImplementedError(
        f"risk distortion function {risk_distortion_type} not implemented")


def cpw_distortion(n: float, tau: torch.Tensor):
    """
    Risk distortion based on cumulative probability weighting
    Referened in IQN paper, based on cumulative prospect theory.
    """
    num = tau**n
    denom = (tau**n + (1 - tau)**n)**1 / n
    return num / denom


def wang_distortion(n: float, tau: torch.Tensor):
    """
    Risk Distortion as per Wang 2002 (referenced in iqn paper).
    Inputs are clipped to prevent numerical instability in the icdf func
    """
    tau = torch.clip(tau, 1e-7, 1. - 1e-7)
    inv = std_normal_dist.icdf(tau) + n
    return std_normal_dist.cdf(inv)


def pow_distortion(n: float, tau: torch.Tensor):
    """ Power formula as in the IQN paper """
    if n < 0:
        return 1 - (1 - tau)**(1 / (1 + abs(n)))
    return tau**(1 / (1 + abs(n)))


def cvar_distortion(n: float, tau: torch.Tensor):
    """ Conditional Value at Risk"""
    return n * tau


class IQN(DQN):
    """
    Implements a base DQN agent from which extensions can inherit
    The Agent instance can be called directly to get an action based on a state:
        action = dqn(state)
    or:
        action = agent.get_action(state)
    use dqn.step(n) to step through n environment interactions
    The method for training a single batch is self.train_step(sarsd) where sarsd is a class with ndarray members (I.e of shape (bs, time, feats))
    """
    def __init__(
            self,
            env,
            preprocessor,
            input_shape: tuple,
            action_space: tuple,
            discount: float,
            nstep_return: int,
            reduce_rewards: bool,
            reward_shaper_config: Config,
            replay_size: int,
            replay_min_size: int,
            prioritized_replay: bool,
            per_alpha: float,
            per_beta: float,
            per_beta_steps: int,
            noisy_net: bool,
            noisy_net_sigma: float,
            eps: float,
            eps_decay: float,
            eps_min: float,
            batch_size: int,
            test_steps: int,
            unit_size: float,
            savepath: Union[Path, str],
            double_dqn: bool,
            tau_soft_update: float,
            model_class: str,
            model_config: Union[dict, Config],
            lr: float,
            ##############
            # Extra 4 Args specific to IQN
            ##############
            nTau1: int,
            nTau2: int,
            k_huber: float,
            risk_distortion: str,
            risk_distortion_param: float):
        super().__init__(env, preprocessor, input_shape, action_space,
                         discount, nstep_return, reduce_rewards,
                         reward_shaper_config, replay_size, replay_min_size,
                         prioritized_replay, per_alpha, per_beta,
                         per_beta_steps, noisy_net, noisy_net_sigma, eps,
                         eps_decay, eps_min, batch_size, test_steps, unit_size,
                         savepath, double_dqn, tau_soft_update, model_class,
                         model_config, lr)

        self.nTau1 = nTau1
        self.nTau2 = nTau2
        self.k_huber = k_huber
        self.risk_distortion = make_risk_distortion(risk_distortion,
                                                    risk_distortion_param)

        # self.desired_port = torch.tensor([1., 0.], device=self.device)[None, :]

    @classmethod
    def from_config(cls, config):
        env = make_env(config)
        preprocessor = make_preprocessor(config, env.nAssets)
        input_shape = preprocessor.feature_output_shape
        atoms = config.discrete_action_atoms + 1
        action_space = DiscreteRangeSpace((0, atoms), env.nAssets)
        aconf = config.agent_config
        unit_size = aconf.unit_size_proportion_avM
        savepath = Path(config.basepath) / config.experiment_id / 'models'
        return cls(env, preprocessor, input_shape, action_space,
                   aconf.discount, aconf.nstep_return, aconf.reduce_rewards,
                   config.reward_shaper_config, aconf.replay_size,
                   aconf.replay_min_size, aconf.prioritized_replay,
                   aconf.per_alpha, aconf.per_beta, aconf.per_beta_steps,
                   aconf.noisy_net, aconf.noisy_net_sigma, aconf.eps,
                   aconf.eps_decay, aconf.eps_min, aconf.batch_size,
                   config.test_steps, unit_size, savepath, aconf.double_dqn,
                   aconf.tau_soft_update, config.model_config.model_class,
                   config.model_config, config.optim_config.lr, aconf.nTau1,
                   aconf.nTau2, aconf.k_huber, aconf.risk_distortion,
                   aconf.risk_distortion_param)

    @torch.no_grad()
    def get_quantiles(self,
                      state,
                      target=False,
                      risk_distort=False,
                      device=None):
        device = device or self.device
        state = self.prep_state_tensors(state, device=device)
        if target:
            return self.model_t(state)
        tau = torch.rand(1,
                         self.nTau1,
                         dtype=torch.float32,
                         device=self.device,
                         requires_grad=False)
        if risk_distort:
            tau = self.risk_distortion(tau)
        return self.model_b(state, tau=tau)

    @torch.no_grad()
    def get_qvals(self, state, target=False, risk_distort=True, device=None):
        """
        External interface - for inference and env interaction
        Takes in numpy arrays
        and return qvals for actions
        """
        quantiles = self.get_quantiles(
            state, target=target, risk_distort=risk_distort,
            device=device)  #(bs, nTau1, n_assets, n_actions)
        return quantiles.mean(1)  #(bs, n_assets, n_actions)

    def __call__(self,
                 state: State,
                 target: bool = True,
                 device: torch.device = None):
        return self.get_action(state, target=target, device=device)

    @torch.no_grad()
    def calculate_Gt_target(self, next_state, reward, done):
        """
        Given a next_state State object, calculates the target value
        to be used in td error and loss calculation
        """
        bs = reward.shape[0]
        tau_greedy = torch.rand(bs,
                                self.nTau1,
                                dtype=torch.float32,
                                device=reward.device,
                                requires_grad=False)
        tau_greedy = self.risk_distortion(tau_greedy)
        tau2 = torch.rand(bs,
                          self.nTau2,
                          dtype=torch.float32,
                          device=reward.device,
                          requires_grad=False)

        if self.double_dqn:
            greedy_quantiles = self.model_b(
                next_state, tau=tau_greedy)  # (bs, nTau1, nassets, nactions)
        else:
            greedy_quantiles = self.model_t(
                next_state, tau=tau_greedy)  # (bs, nTau1, nassets, nactions)
        greedy_actions = torch.argmax(greedy_quantiles.mean(1),
                                      dim=-1,
                                      keepdim=True)  # (bs, nassets,  nactions)
        assert greedy_actions.shape[1:] == (self.n_assets, 1)
        one_hot = F.one_hot(greedy_actions,
                            self.action_atoms).to(reward.device)
        quantiles_next = self.model_t(next_state, tau=tau2)
        assert quantiles_next.shape[1:] == (self.nTau2, self.n_assets,
                                            self.action_atoms)
        # quantiles_next = (
        #     quantiles_next * one_hot[:, None, :, 0, :]).sum(-1).mean(
        #         -1)  # get max qval within asset and average across assets
        # assert quantiles_next.shape[1:] == (self.nTau2, )
        # Gt = reward[:, None] + (~done[:, None] *
        #                         (self.discount**self.nstep_return) *
        #                         quantiles_next)
        # assert Gt.shape[1:] == (self.nTau2, )
        # PARALLEL REWARDS VERSION
        quantiles_next = (quantiles_next * one_hot[:, None, :, 0, :]).sum(-1)
        assert quantiles_next.shape[1:] == (self.nTau2, self.n_assets)
        Gt = reward[:, None, :] + (~done[:, None, None] *
                                   (self.discount**self.nstep_return) *
                                   quantiles_next)
        assert Gt.shape[1:] == (self.nTau2, self.n_assets)
        return Gt

    def train_step(self, sarsd: SARSD = None, weights: np.ndarray = None):
        self.model_b.sample_noise()
        self.model_t.sample_noise()
        sarsd, weights = self.buffer.sample(
            self.batch_size) if sarsd is None else (sarsd, weights)
        state, action, reward, next_state, done = self.prep_sarsd_tensors(
            sarsd)
        bs = reward.shape[0]
        tau1 = torch.rand(bs,
                          self.nTau1,
                          dtype=torch.float32,
                          device=reward.device)
        quantiles = self.model_b(state, tau=tau1)
        action_mask = F.one_hot(action[:, None],
                                self.action_atoms).to(self.device)

        Gt = self.calculate_Gt_target(next_state, reward, done)  # (bs, nTau2)
        # Qt = (quantiles * action_mask).sum(-1).mean(-1)  # (bs, nTau1)
        # PARALLEL REWARDS VERSION
        Qt = (quantiles * action_mask).sum(-1)  # (bs, nTau1, self.n_assets)
        if self.prioritized_replay:
            weights = torch.from_numpy(weights).to(self.device)
            loss, td_error = self.loss_fn(Qt, Gt, tau1, weights)
            self.buffer.update_priority(td_error)
        else:
            loss, td_error = self.loss_fn(Qt, Gt, tau1, None)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model_b.parameters(),
                                 max_norm=1.,
                                 norm_type=2)
        self.opt.step()

        self.update_target()
        return {
            'loss': loss.detach().item(),
            'td_error': td_error.mean().item(),
            'Qt': Qt.detach().mean().item(),
            'Gt': Gt.detach().mean().item()
        }

    def loss_fn(self, *args, **kwargs):
        return self.loss_fn_huber(*args, **kwargs)

    def loss_fn_huber(self, Qt, Gt, tau, weights: torch.Tensor = None):
        """
        Quantile Huber Loss
        returns:  (loss, td_error)
            loss: scalar
            td_error: scalar
        """
        # assert Qt.shape[1:] == (self.nTau1, )
        # assert Gt.shape[1:] == (self.nTau2, )
        # PARALLEL REWARDS VERSION
        assert Qt.shape[1:] == (self.nTau1, self.n_assets)
        assert Gt.shape[1:] == (self.nTau2, self.n_assets)
        td_error = Gt.unsqueeze(1) - Qt.unsqueeze(2)
        # assert td_error.shape[1:] == (
        #     self.nTau1,
        #     self.nTau2,
        # )
        # PARALLEL REWARDS VERSION
        assert td_error.shape[1:] == (self.nTau1, self.nTau2, self.n_assets)
        huber_loss = torch.where(
            td_error.abs() <= self.k_huber, 0.5 * td_error.pow(2),
            self.k_huber * (td_error.abs() - self.k_huber / 2))
        assert huber_loss.shape == td_error.shape
        # quantile_loss = torch.abs(tau[:, :, None] -
        #                           (td_error.detach() < 0.).float()) *\
        #     huber_loss / self.k_huber
        # PARALLEL REWARDS
        quantile_loss = torch.abs(tau[:, :, None, None] -
                                  (td_error.detach() < 0.).float()) *\
            huber_loss / self.k_huber
        assert quantile_loss.shape == huber_loss.shape
        if weights is None:
            # loss = quantile_loss.mean(-1).sum(-1)
            # PARALLEL REWARDS
            loss = quantile_loss.sum(-1).mean(-1).sum(-1)
        else:
            # loss = quantile_loss.mean(-1).sum(-1) * weights
            # PARALLEL REWARD
            loss = quantile_loss.sum(-1).mean(-1).sum(-1) * weights
        # assert loss.shape == (Qt.shape[0], )
        #PARALLEL REWARDS
        assert loss.shape == (Qt.shape[0], )
        return loss.mean(), td_error.abs().mean((-1, -2, -3)).detach()

    def loss_fn_mse(self, Qt, Gt, tau, weights: torch.Tensor = None):
        """
        Quantile MSE Loss
        returns:  (loss, td_error)
            loss: scalar
            td_error: scalar
        """
        assert Qt.shape[1:] == (self.nTau1, )
        assert Gt.shape[1:] == (self.nTau2, )
        td_error = Gt.unsqueeze(1) - Qt.unsqueeze(2)
        assert td_error.shape[1:] == (
            self.nTau1,
            self.nTau2,
        )
        mse_loss = 0.5 * td_error.pow(2)
        assert mse_loss.shape == td_error.shape
        quantile_loss = (torch.abs(tau[:, :, None] -
                                   (td_error.detach() < 0.).float()) *
                         mse_loss)
        assert quantile_loss.shape == mse_loss.shape
        if weights is None:
            loss = quantile_loss.mean(-1).sum(-1)
        else:
            loss = quantile_loss.mean(-1).sum(-1) * weights
        assert loss.shape == (Qt.shape[0], )
        return loss.mean(), td_error.abs().mean(-1).mean(-1).detach()

    @torch.no_grad()
    def test_episode(self, test_steps=None, reset=True, target=True) -> dict:
        self.test_mode()
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
            qvals = self.get_qvals(state, target=target, risk_distort=True)
            action = self.get_action(qvals=qvals, target=target).cpu().numpy()
            transaction = self.action_to_transaction(action)
            _tst_metrics['timestamp'] = state.timestamp[-1]
            state, reward, done, info = self._env.step(transaction)
            self._preprocessor.stream_state(state)
            state = self._preprocessor.current_data()
            _tst_metrics['qvals'] = qvals.cpu().numpy()
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


class IQNCURL(IQN):
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


# for temporary backward comp
IQNReverser = IQN


class IQNMixedActions(IQN):
    def __init__(
            self,
            env,
            preprocessor,
            input_shape: tuple,
            action_space: tuple,
            discount: float,
            nstep_return: int,
            reduce_rewards: bool,
            reward_shaper_config: Config,
            replay_size: int,
            replay_min_size: int,
            prioritized_replay: bool,
            per_alpha: float,
            per_beta: float,
            per_beta_steps: int,
            noisy_net: bool,
            noisy_net_sigma: float,
            eps: float,
            eps_decay: float,
            eps_min: float,
            batch_size: int,
            test_steps: int,
            unit_size: float,
            savepath: Union[Path, str],
            double_dqn: bool,
            tau_soft_update: float,
            model_class: str,
            model_config: Union[dict, Config],
            lr: float,
            ##############
            # Extra 4 Args specific to IQN
            ##############
            nTau1: int,
            nTau2: int,
            k_huber: float,
            risk_distortion: str,
            risk_distortion_param: float):
        DQNMixedActions.__init__(
            self, env, preprocessor, input_shape, action_space, discount,
            nstep_return, reduce_rewards, reward_shaper_config, replay_size,
            replay_min_size, prioritized_replay, per_alpha, per_beta,
            per_beta_steps, noisy_net, noisy_net_sigma, eps, eps_decay,
            eps_min, batch_size, test_steps, unit_size, savepath, double_dqn,
            tau_soft_update, model_class, model_config, lr)

        self.nTau1 = nTau1
        self.nTau2 = nTau2
        self.k_huber = k_huber
        self.risk_distortion = make_risk_distortion(risk_distortion,
                                                    risk_distortion_param)

        # self.desired_port = torch.tensor([1., 0.], device=self.device)[None, :]

    @classmethod
    def from_config(cls, config):
        env = make_env(config)
        preprocessor = make_preprocessor(config, env.nAssets)
        input_shape = preprocessor.feature_output_shape
        atoms = config.discrete_action_atoms + 1
        # Parallel Actions for reference ####################################
        # action_space = DiscreteRangeSpace((0, atoms), env.nAssets)
        # ALL MIXED ACTIONS - Full product between n_assets*action_atoms #####
        # atoms = atoms ** env.nAssets
        action_space = DiscreteRangeSpace((0, atoms), 1)
        aconf = config.agent_config
        unit_size = aconf.unit_size_proportion_avM
        savepath = Path(config.basepath) / config.experiment_id / 'models'
        return cls(env, preprocessor, input_shape, action_space,
                   aconf.discount, aconf.nstep_return, aconf.reduce_rewards,
                   config.reward_shaper_config, aconf.replay_size,
                   aconf.replay_min_size, aconf.prioritized_replay,
                   aconf.per_alpha, aconf.per_beta, aconf.per_beta_steps,
                   aconf.noisy_net, aconf.noisy_net_sigma, aconf.eps,
                   aconf.eps_decay, aconf.eps_min, aconf.batch_size,
                   config.test_steps, unit_size, savepath, aconf.double_dqn,
                   aconf.tau_soft_update, config.model_config.model_class,
                   config.model_config, config.optim_config.lr, aconf.nTau1,
                   aconf.nTau2, aconf.k_huber, aconf.risk_distortion,
                   aconf.risk_distortion_param)

    def action_to_transaction(
            self, action: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        return DQNMixedActions.action_to_transaction(self, action)
