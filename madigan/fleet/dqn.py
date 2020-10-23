import os
from typing import Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .agent import Agent
from .conv_model import ConvModel
from .mlp_model import MLPModel
from ..utils import default_device, DiscreteActionSpace
from ..utils.config import Config
from ..utils.data import State

def get_model_class(name):
    if name == "ConvModel":
        return ConvModel
    elif name == "MLPModel":
        return MLPModel
    else:
        raise NotImplementedError(f"model {name} is not Implemented")

# p = type('params', (object, ), params)

class DQN(Agent):
    """
    Implements a base DQN agent from which extensions can inherit
    The Agent instance can be called directly to get an action based on a state:
        action = dqn(state)
    The method for training a single batch is self.train_step(sarsd) where sarsd is a class with ndarray members (I.e of shape (bs, time, feats))
    """
    def __init__(self, config, env, name=None, device=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = Config(config)
        self._env = env
        a_config = config.agent_config
        m_config = a_config.model_config
        o_config = a_config.optim_config
        self.double_dqn = a_config['double_dqn']
        self.discount = a_config.discount
        self.action_atoms = m_config.action_atoms

        self.lot_unit_value = m_config.lot_unit_value
        actions = [self.lot_unit_value*action - self.action_atoms//2
                   for action in range(self.action_atoms)]
        probs = [1/len(actions) for i in actions]
        self._action_space = DiscreteActionSpace(actions, probs, len(config.assets))
        self.min_tf = config.min_tf
        self.savepath = Path(config.basepath)/f'{config.experiment_id}/models'
        if not self.savepath.is_dir():
            self.savepath.mkdir(parents=True)
        self.model_class = get_model_class(m_config['model_class'])

        self.model_b = self.model_class(**m_config).to(device)
        self.model_t = self.model_class(**m_config).to(device)
        self.model_t.eval()

        if o_config['type'] == 'Adam':
            self.opt = torch.optim.Adam(self.model_b.parameters(), lr=o_config.lr)
        else:
            raise ValueError("Only 'Adam' accepted as type of optimizer in config")

        self.reward_clip = config.reward_clip or None
        # SCHEDULER NOT YET IN USE
        USE_SCHED=False
        if USE_SCHED:
            self.lr_sched = torch.optim.lr_scheduler.CyclicLR(self.opt, base_lr=o_config.lr,
                                                              max_lr=1e-2, step_size_up=2000,
                                                              mode="exp_range")
        else:
            # Dummy class for now
            class Sched:
                def step(self): pass
            self.lr_sched = Sched()

        # Overwrites previously saved models
        if config.overwrite_exp:
            print('deleting previously saved models')
            self._delete_models()

        saved_models = list(self.savepath.iterdir())
        if len(saved_models):
            print('loading latest')
            self.load_latest_state() # tries to load latest model by default
        else:
            self.training_steps = 0
            self.total_steps = 0
        super().__init__(name)

    @property
    def env(self):
        """
        Reference to current env - useful for current prices/availablae Margin etc
        Be careful - don't mess with it aside from accessing properties
        """
        return self._env

    @property
    def action_space(self):
        """ Action space object which can be sampled from"""
        return self._action_space

    def target_update_hard(self):
        """ Hard update, copies weights """
        self.model_t.load_state_dict(self.model_b.state_dict())

    def target_update_soft(self):
        """ Incremental 'soft' update of target proportional to tau_soft parameter (I.e 1e-4)"""
        for behaviour, target in zip(self.model_b.parameters(), self.model_t.parameters()):
            target.data.copy_(self.tau_soft * behaviour.data + (1.-self.tau_soft)*target.data)

    def save_state(self):
        self.save_checkpoint("main")

    def save_checkpoint(self, name):
        config = {'state_dict': self.model_b.state_dict(),
                  'training_steps': self.training_steps,
                  'total_steps': self.total_steps}
        torch.save(config, self.savepath/f'{name}_{self.total_steps}.pth')

    def _delete_models(self):
        if self.config.overwrite_exp:
            saved_models = list(self.savepath.iterdir())
            if len(saved_models):
                for model in saved_models:
                    os.remove(model)
        else:
            raise NotImplementedError("Attempting to delete models when config.overwrite_exp is not set to true")

    def get_latest_state(self):
        model_savepoints = [(int(name.stem[5:]), name) for name in self.savepath.iterdir() if "main" in name.stem]
        if len(model_savepoints) > 0 :
            model_savepoints.sort(reverse=True)
            return model_savepoints[0][1]
        else: return None

    def load_latest_state(self):
        self.load_state(self.get_latest_state())

    def load_state(self, savepoint):
        if savepoint is not None:
            state = torch.load(savepoint)
            self.model_b.load_state_dict(state['state_dict'])
            self.model_t.load_state_dict(state['state_dict'])
            self.training_steps = state['training_steps']
            self.total_steps = state['total_steps']

    def actions_to_transactions(self, actions):
        transactions = actions - self.action_atoms // 2
        transactions *= self.lot_unit_value
        # transactions = transactions // prices
        return transactions

    def transactions_to_actions(self, transactions):
        # actions = np.rint((prices*transactions) // self.lot_unit_value) + (self.action_atoms//2)
        actions = np.rint(transactions // self.lot_unit_value) + (self.action_atoms//2)
        return actions

    def qvals_to_transactions(self, qvals, boltzmann=False, boltzmann_temp=1.):
        if boltzmann:
            distribution = torch.distributions.Categorical(logits=qvals/boltzmann_temp)
            actions = distribution.sample()
        else:
            actions = qvals.max(-1)[1]
        actions -= self.action_atoms // 2
        actions *= self.lot_unit_value
        actions = actions.detach().cpu().numpy()
        return actions.astype(np.int64)

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

    def make_state_tensor(self, state, device):
        if len(state.price.shape) == 2:
            price = state.price[None, ...]
            price = torch.tensor(price, dtype=torch.float32, device=device)
        else:
            price = torch.tensor(state.price, dtype=torch.float32, device=device)
        port = torch.tensor(np.atleast_2d(state.portfolio), dtype=torch.float32, device=device)
        return State(price, port, state.timestamp)

    def make_sarsd_tensors(self, sarsd, device=None):
        device = device or default_device()
        price = torch.as_tensor(sarsd.state.price, dtype=torch.float32, device=device)
        port = torch.as_tensor(sarsd.state.portfolio, dtype=torch.float32, device=device)
        state = State(price, port, sarsd.state.timestamp)
        # action_model = (np.rint((sarsd.state.price[:, -1, :] *sarsd.action) // self.lot_unit_value)\
        #                 + (self.action_atoms//2))
        action_model = (np.rint(sarsd.action // self.lot_unit_value)\
                        + (self.action_atoms//2))
        action = torch.as_tensor((action_model), dtype=torch.long, device=device)
        reward = torch.as_tensor(sarsd.reward, dtype=torch.float32, device=device)
        next_price = torch.as_tensor(sarsd.next_state.price, dtype=torch.float32, device=device)
        next_port = torch.as_tensor(sarsd.next_state.portfolio, dtype=torch.float32, device=device)
        next_state = State(next_price, next_port, sarsd.next_state.timestamp)
        done = ~torch.as_tensor(sarsd.done, dtype=torch.bool, device=device)
        return state, action, reward, next_state, done

    def get_qvals(self, state, target=True, device=None):
        """
        External interface - for inference
        Takes in numpy arrays
        and return qvals for actions
        """
        device = device or default_device()
        state = self.make_state_tensor(state, device=device)
        with torch.no_grad():
            if target:
                return self.model_t(state)
            return self.model_b(state)

    def __call__(self, state: State, target: bool=True, raw_qvals: bool=False,
                 max_qvals: bool=False, boltzmann: bool=False, boltzmann_temp: float = 1.):
        """
        External interface - for inference
        takes in numpy arrays and returns greedy actions
        """
        # price = state.price[None, ...]
        # port = state.portfolio[None, ...]
        qvals = self.get_qvals(state, target=target)
        if raw_qvals:
            return qvals
        # max_qvals = qvals.max(-1)[0]
        transactions = self.qvals_to_transactions(qvals, boltzmann=boltzmann,
                                                  boltzmann_temp=boltzmann_temp)
        return self.filter_transactions(transactions[0], state.portfolio)

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

