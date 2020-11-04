import os
from typing import Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .base import OffPolicyQ
from .dqn import DQN
from ...environments import make_env
from ..net.conv_net_iqn import ConvNetIQN
# from ..net.mlp_net import MLPNet
from ...utils import default_device, DiscreteActionSpace, DiscreteRangeSpace, ternarize_array
from ...utils.preprocessor import make_preprocessor
from ...utils.config import Config
from ...utils.data import State

def get_model_class(name):
    if name in ("ConvNet", ):
        return ConvNetIQN
    # elif name == ("MLPNet", ):
    #     return MLPNetIQN
    else:
        raise NotImplementedError(f"model {name} is not Implemented")

# p = type('params', (object, ), params)

class IQN(DQN):
    """
    Implements a base DQN agent from which extensions can inherit
    The Agent instance can be called directly to get an action based on a state:
        action = dqn(state)
    or:
        action = dqn.get_action(state)
    use dqn.step(n) to step through n environment interactions
    The method for training a single batch is self.train_step(sarsd) where sarsd is a class with ndarray members (I.e of shape (bs, time, feats))
    """
    def __init__(self,
                 env,
                 preprocessor,
                 input_shape: tuple,
                 action_space: tuple,
                 discount: float,
                 nstep_return: int,
                 replay_size: int,
                 replay_min_size: int,
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
                 # Extra 3 Args
                 ##############
                 nTau1: int,
                 nTau2: int,
                 k_huber: float):
        super().__init__(env, preprocessor, input_shape, action_space,  discount,
                       nstep_return, replay_size, replay_min_size, eps, eps_decay,
                       eps_min, batch_size, test_steps, unit_size, savepath,
                       double_dqn, tau_soft_update, model_class, model_config, lr)

        self.nTau1 = nTau1
        self.nTau2 = nTau2
        self.k_huber = k_huber
        self.risk_distortion = lambda x: x

    @classmethod
    def from_config(cls, config):
        env = make_env(config)
        preprocessor = make_preprocessor(config)
        input_shape = preprocessor.feature_output_shape
        atoms = config.discrete_action_atoms
        action_space = DiscreteRangeSpace((-atoms//2, atoms//2 + 1),
                                          config.n_assets)
        aconf = config.agent_config
        unit_size = aconf.unit_size_proportion_avM
        return cls(env, preprocessor, input_shape, action_space, aconf.discount,
                   aconf.nstep_return, aconf.replay_size, aconf.replay_min_size,
                   aconf.eps, aconf.eps_decay, aconf.eps_min, aconf.batch_size,
                   config.test_steps, unit_size, Path(config.experiment_path)/'models',
                   aconf.double_dqn, aconf.tau_soft_update,
                   config.model_config.model_class, config.model_config,
                   config.optim_config.lr, config.agent_config.nTau1,
                   config.agent_config.nTau2, config.agent_config.k_huber
                   )

    @torch.no_grad()
    def get_quantiles(self, state, target=False, device=None):
        device = device or self.device
        state = self.prep_state_tensors(state, device=device)
        if target:
            return self.model_t(state)
        return self.model_b(state)

    @torch.no_grad()
    def get_qvals(self, state, target=False, device=None):
        """
        External interface - for inference and env interaction
        Takes in numpy arrays
        and return qvals for actions
        """
        quantiles = self.get_quantiles(state, target=target, device=device) #(bs, nTau1, n_assets, n_actions)
        return quantiles.mean(1) #(bs, n_assets, n_actions)

    def __call__(self, state: State, target: bool = True, device: torch.device=None):
        return self.get_action(state, target=target, device=device)

    @torch.no_grad()
    def calculate_Gt_target(self, next_state, reward, done):
        """
        Given a next_state State object, calculates the target value
        to be used in td error and loss calculation
        """
        bs = reward.shape[0]
        tau_greedy = torch.rand(bs, self.nTau1, dtype=torch.float32,
                                device=reward.device, requires_grad=False)
        tau_greedy = self.risk_distortion(tau_greedy)
        tau2 = torch.rand(bs, self.nTau2, dtype=torch.float32,
                                device=reward.device, requires_grad=False)

        if self.double_dqn:
            greedy_quantiles = self.model_b(next_state, tau=tau_greedy) #(bs, nTau1, nassets, nactions)
        else:
            greedy_quantiles = self.model_t(next_state, tau=tau_greedy) #(bs, nTau1, nassets, nactions)
        greedy_actions = torch.argmax(greedy_quantiles.mean(1), dim=-1, keepdim=True) #(bs, nassets,  nactions)
        assert greedy_actions.shape[1:] == (self.n_assets, 1)
        one_hot = F.one_hot(greedy_actions, self.action_atoms).to(reward.device)
        quantiles_next = self.model_t(next_state, tau=tau2)
        assert quantiles_next.shape[1:] == (self.nTau2, self.n_assets, self.action_atoms)
        quantiles_next = (quantiles_next * one_hot).sum(-1)
        assert quantiles_next.shape[1:] == (self.nTau2, self.n_assets)
        Gt = reward[:, None, None] + (~done[:, None, None] *\
                                (self.discount ** self.nstep_return) *\
                                quantiles_next)
        assert Gt.shape[1:] == (self.nTau2, self.n_assets)
        return Gt

    def train_step(self, sarsd=None):
        sarsd = sarsd or self.buffer.sample(self.batch_size)
        state, action, reward, next_state, done = self.prep_sarsd_tensors(sarsd)
        bs = reward.shape[0]
        tau1 = torch.rand(bs, self.nTau1, dtype=torch.float32,
                          device=reward.device)
        quantiles = self.model_b(state, tau=tau1)
        action_mask = F.one_hot(action[:, None], self.action_atoms).to(self.device)

        Gt = self.calculate_Gt_target(next_state, reward, done)
        Qt = (quantiles*action_mask).sum(-1)
        # assert Qt.shape == Gt.shape
        loss, td_error = self.loss_fn(Qt, Gt, tau1)
        self.opt.zero_grad()
        loss.backward()
        # with torch.no_grad():
        #     total_norm = 0.
        #     for p in self.model_b.parameters():
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item()**2
        #     total_norm = total_norm ** (1./2)
        #     if total_norm >= .5:
        #         print(total_norm)
        torch.nn.utils.clip_grad_norm_(self.model_b.parameters(), max_norm=1.,
                                       norm_type=2)
        self.opt.step()

        self.update_target()
        return {'loss': loss.detach().item(), 'td_error': td_error.detach().item(),
                'Qt': Qt.detach().mean().item(), 'Gt': Gt.detach().mean().item()}

    def loss_fn(self, Qt, Gt, tau):
        """
        Quantile Huber Loss
        returns:  (loss, td_error)
            loss: scalar
            td_error: scalar
        """
        assert Qt.shape[1:] == (self.nTau1, self.n_assets)
        assert Gt.shape[1:] == (self.nTau2, self.n_assets)
        td_error = Gt.unsqueeze(1) - Qt.unsqueeze(-1)
        assert td_error.shape[1:] == (self.nTau1, self.nTau2, self.n_assets)
        huber_loss = torch.where(td_error.abs() <= self.k_huber,
                                 0.5*td_error.pow(2),
                                 self.k_huber * (td_error.abs() - self.k_huber/2))
        assert huber_loss.shape == td_error.shape
        quantile_loss = torch.abs(tau[..., None, None] - (td_error.detach() < 0.).float()) *\
            huber_loss / self.k_huber
        assert quantile_loss.shape == huber_loss.shape

        return quantile_loss.mean(-1).sum(-1).mean(), td_error.abs().mean()
