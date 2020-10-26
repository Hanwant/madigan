import os
from typing import Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .base import OffPolicyQ
from ...environments import make_env
from ..net.conv_model import ConvModel
from ..net.mlp_model import MLPModel
from ...utils import default_device, DiscreteActionSpace, DiscreteRangeSpace, ternarize_array
from ...utils.preprocessor import make_preprocessor
from ...utils.config import Config
from ...utils.data import State

def get_model_class(name):
    if name == "ConvModel":
        return ConvModel
    elif name == "MLPModel":
        return MLPModel
    else:
        raise NotImplementedError(f"model {name} is not Implemented")

# p = type('params', (object, ), params)

class DQN(OffPolicyQ):
    """
    Implements a base DQN agent from which extensions can inherit
    The Agent instance can be called directly to get an action based on a state:
        action = dqn(state)
    The method for training a single batch is self.train_step(sarsd) where sarsd is a class with ndarray members (I.e of shape (bs, time, feats))
    """
    def __init__(self, env,
                 input_space: tuple,
                 action_space: tuple,
                 discount: float,
                 nstep_return,
                 replay_size,
                 replay_min_size,
                 batch_size,
                 double_dqn: bool,
                 model_class: str,
                 model_config: Union[dict, Config],
                 savepath: Union[Path, str], lr=1e-3):
        super().__init__(env, input_space, action_space, discount, nstep_return,
                         replay_size, replay_min_size)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._action_space = DiscreteRangeSpace(-self.action_atoms//2, self.action_atoms//2 + 1)
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.discount = discount
        self.model_class = get_model_class(model_class)
        self.model_b = self.model_class(input_space, self.action_space, **model_config)
        self.model_t = self.model_class(input_space, self.action_space, **model_config)
        self.model_t.load_state_dict(self.model_b.state_dict())
        self.opt = torch.optim.Adam(self.model_b.parameters(), lr=lr)
        self.savepath = savepath

        self.training_steps = 0
        self.env_steps = 0

        # SCHEDULER NOT YET IN USE
        USE_SCHED=False
        if USE_SCHED:
            self.lr_sched = torch.optim.lr_scheduler.CyclicLR(self.opt, base_lr=o_config.lr,
                                                              max_lr=1e-2, step_size_up=2000)
        else:
            # Dummy class for now
            class Sched:
                def step(self): pass
            self.lr_sched = Sched()

    @classmethod
    def from_config(cls, config):
        env = make_env(config)
        preprocessor = make_preprocessor(config)
        input_space = preprocessor.feature_output_shape
        action_space = (config.n_assets, config.discrete_action_atoms)
        aconf = config.agent_config
        return cls(env, preprocessor, input_space, action_space, aconf.discount,
                   aconf.nstep_return, aconf.replay_size, aconf.replay_min_size,
                   aconf.batch_size, aconf.double_dqn, aconf.model_config.model_class,
                   Path(config.basepath)/'models')

    @property
    def env(self):
        """
        Reference to current env - useful for current prices/availablae Margin etc
        Be careful - don't mess with it aside from accessing properties
        """
        return self._env

    def to(self, device):
        self.device = torch.device(device)
        self.model_b.to(self.device)
        self.model_t.to(self.device)

    @property
    def action_space(self):
        """ Action space object which can be sampled from"""
        units = 0.05 * self.env.availableMargin / self.env.currentPrices
        self._action_space.action_multiplier = units
        return self._action_space


    def actions_to_transactions(self, actions):
        transactions = actions - self.action_atoms // 2
        transactions *= self.lot_unit_value
        # transactions = transactions // prices
        return transactions

    def transactions_to_actions(self, transactions):
        # actions = np.rint((prices*transactions) // self.lot_unit_value) + (self.action_atoms//2)
        actions = ternarize_array(transactions) + self.action_atoms // 2
        return actions

    @torch.no_grad()
    def get_qvals(self, state, target=True, device=None):
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
    def get_actions(self, state, target=True, device=None):
        """
        External interface - for inference and env interaction
        takes in numpy arrays and returns greedy actions
        """
        qvals = self.get_qvals(state, target=target, device=device)
        actions = qvals.max(-1)[0]
        return self.actions_to_transactions(actions)


    def __call__(self, state: State, target: bool = True, raw_qvals: bool = False,
                 max_qvals: bool = False):
        return self.get_actions(state, target=target)

    def filter_transactions(self, transactions, portfolio):
        """
        Doesn't allow doubling up on positions
        """
        for i, action in enumerate(transactions):
            if portfolio[i] == 0.:
                pass
            elif np.sign(portfolio[i]) == np.sign(action):
                    transactions[i] = 0.
        return transactions

    def prep_state_tensors(self, state, batch=False, device=None):
        if not batch:
            price = torch.as_tensor(state.price[None, ...], dtype=torch.float32).to(self.device)
            port = torch.as_tensor(state.portfolio[None, -1], dtype=torch.float32).to(self.device)
        else:
            price = torch.as_tensor(state.price, dtype=torch.float32).to(self.device)
            port = torch.as_tensor(state.portfolio[:, -1], dtype=torch.float32).to(self.device)
#         timestamp = torch.as_tensor(state.timestamp)
        return State(price, port, state.timestamp)

    def prep_sarsd_tensors(self, sarsd, device=None):
        state = self.prep_state(sarsd.state, batch=True)
#         action = np.rint(sarsd.action // self.lot_unit_value) + self.action_atoms//2
        action = self.transaction_to_actions(sarsd.action)
        action = torch.as_tensor(action, dtype=torch.long, device=self.device)#[..., 0]
        reward = torch.as_tensor(sarsd.reward, dtype=torch.float32, device=self.device)
        next_state = self.prep_state(sarsd.next_state, batch=True)
        done = torch.as_tensor(sarsd.done, dtype=torch.bool, device=self.device)
        return state, action, reward, next_state, done

    def loss(self, Q_t, G_t):
        return F.smooth_l1_loss(Q_t, G_t)

    def train_step(self, sarsd):
        states, actions, rewards, next_states, done_mask = self.make_sarsd_tensors(sarsd)
        if self.reward_clip is not None:
            rewards = rewards.clamp(min=self.reward_clip[0], max=self.reward_clip[1])
        self.opt.zero_grad()
        with torch.no_grad():
            if self.double_dqn:
                b_actions = self.model_b(next_states).max(-1)[1]
                b_actions_mask = F.one_hot(b_actions, self.action_atoms).to(b_actions.device)
                qvals_next = self.model_t(next_states)
                greedy_qvals_next = torch.sum(qvals_next * b_actions_mask, -1)
            else:
                greedy_qvals_next = self.model_t(next_states).max(-1)[0]
            G_t = rewards[..., None] + (done_mask[..., None] * self.discount * greedy_qvals_next)
        actions_mask = F.one_hot(actions, self.action_atoms).to(actions.device)
        qvals = self.model_b(states)
        Q_t = (qvals * actions_mask).sum(-1)
        td_error = (G_t - Q_t).mean()
        loss = self.loss(Q_t, G_t)
        loss.backward()
        self.opt.step()
        self.lr_sched.step()
        return {'loss': loss.detach().item(), 'td_error': td_error.detach().item(), 'G_t': G_t.detach(), 'Q_t': Q_t.detach()}
        # return loss.detach().item(), td_error, G_t.detach(), Q_t.detach()

    def target_update_hard(self):
        """ Hard update, copies weights """
        self.model_t.load_state_dict(self.model_b.state_dict())

    def target_update_soft(self):
        """ Incremental 'soft' update of target proportional to tau_soft parameter (I.e 1e-4)"""
        for behaviour, target in zip(self.model_b.parameters(), self.model_t.parameters()):
            target.data.copy_(self.tau_soft * behaviour.data + (1.-self.tau_soft)*target.data)

    def save_state(self, branch=None):
        branch = branch or "main"
        self.save_checkpoint("main")
        state = {'state_dict': self.model_b.state_dict(),
                  'training_steps': self.training_steps,
                  'env_steps': self.env_steps}
        torch.save(state, self.savepath/f'{branch}.pth')

    def load_state(self, branch=None):
        branch = branch or "main"
        state = torch.load(self.savepath/f'{branch}.pth')
        self.model_b.load_state_dict(state['state_dict'])
        self.model_t.load_state_dict(state['state_dict'])
        self.training_steps = state['training_steps']
        self.env_steps = state['env_steps']

    def _delete_models(self):
        if self.config.overwrite_exp:
            saved_models = list(self.savepath.iterdir())
            if len(saved_models):
                for model in saved_models:
                    os.remove(model)
        else:
            raise NotImplementedError("Attempting to delete models when config.overwrite_exp is not set to true")
