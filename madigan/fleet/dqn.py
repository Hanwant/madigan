from pathlib import Path
import torch
import torch.nn.functional as F
from .agent import Agent
from .conv_model import ConvModel
from .mlp_model import MLPModel
from ..utils import default_device, Config, State

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
    def __init__(self, config, name=None, device=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = Config(config)
        a_config = config.agent_config
        m_config = a_config.model_config
        o_config = a_config.optim_config
        self.savepath = a_config.savepath
        self.double_dqn = a_config['double_dqn']
        self.discount = a_config.discount
        self.nstep = a_config.nstep
        self.action_atoms = a_config.action_atoms

        if Path(self.savepath).is_file():
            config = torch.load(self.savepath)
            self.model_class = get_model_class(m_config['model_class'])
        else:
            self.model_class = get_model_class(m_config['model_class'])

        self.model_b = self.model_class(**m_config).to(device)
        self.model_t = self.model_class(**m_config).to(device)
        self.model_t.eval()
        if o_config['type'] == 'Adam':
            self.opt = torch.optim.Adam(self.model_b.parameters(), lr=o_config.lr,
                                        eps=o_config.eps, betas=o_config.betas)
            # self.opt = torch.optim.Adam(self.model_b.parameters(), lr=o_config.lr)
        else:
            raise ValueError("Only 'Adam' accepted as type of optimizer in config")

        # SCHEDULER NOT YET IN USE
        use_sched=False
        if use_sched:
            self.lr_sched = torch.optim.lr_scheduler.CyclicLR(self.opt, base_lr=o_config.lr, max_lr=1e-2,
                                                              step_size_up=2000, mode="exp_range")
        else:
            # Dummy class for now
            class Sched:
                def step(self): pass
            self.lr_sched = Sched()

        if Path(self.savepath).is_file():
            self.load_state()
        else:
            # self.save_state()
            self.training_steps = 0
            self.total_steps = 0
        super().__init__(name)

    def make_state_tensor(self, state, device):
        price = torch.tensor(state.price, dtype=torch.float32, device=device)
        port = torch.tensor(state.port, dtype=torch.float32, device=device)
        return State(price, port)

    def make_sarsd_tensors(self, sarsd, device=None):
        device = device or default_device()
        price = torch.as_tensor(sarsd.state.price, dtype=torch.float32, device=device)
        port = torch.as_tensor(sarsd.state.port, dtype=torch.float32, device=device)
        state = State(price, port)
        action = torch.as_tensor(sarsd.action, dtype=torch.long, device=device)
        reward = torch.as_tensor(sarsd.reward, dtype=torch.float32, device=device)
        next_price = torch.as_tensor(sarsd.next_state.price, dtype=torch.float32, device=device)
        next_port = torch.as_tensor(sarsd.next_state.port, dtype=torch.float32, device=device)
        next_state = State(next_price, next_port)
        done = ~torch.as_tensor(sarsd.done, dtype=torch.bool, device=device)
        return state, action, reward, next_state, done

    def __call__(self, state, target=True, raw_qvals=False, max_qvals=False):
        """
        External interface - for inference
        takes in numpy arrays and returns greedy actions
        """
        qvals = self.get_qvals(state, target=target)
        if raw_qvals:
            return qvals
        # max_qvals = qvals.max(-1)[0]
        actions = qvals.max(-1)[1]
        return actions

    @torch.no_grad()
    def get_qvals(self, state, target=True, device=None):
        """
        External interface - for inference
        Takes in numpy arrays
        and return qvals for actions
        """
        device = device or default_device()
        state = self.make_state_tensor(state, device=device)
        if target:
            return self.model_t(state)
        return self.model_b(state)

    def loss(self, Q_t, G_t):
        return F.smooth_l1_loss(Q_t, G_t)

    def train_step(self, sarsd):
        states, actions, rewards, next_states, done_mask = self.make_sarsd_tensors(sarsd)
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
        Q_t = (qvals * actions_mask.unsqueeze(1)).sum(-1)
        td_error = (G_t - Q_t).mean().detach().item()
        loss = self.loss(Q_t, G_t)
        loss.backward()
        self.opt.step()
        self.lr_sched.step()
        return {'loss': loss.detach().item(), 'td_error': td_error, 'G_t': G_t, 'Q_t': Q_t}

    def save_state(self):
        config = {'state_dict': self.model_t.state_dict(),
                  'training_steps': self.training_steps,
                  'total_steps': self.total_steps}
        torch.save(config, self.savepath)

    def load_state(self):
        state = torch.load(self.savepath)
        self.model_b.load_state_dict(state['state_dict'])
        self.model_t.load_state_dict(state['state_dict'])
        self.training_steps = state['training_steps']
        self.total_steps = state['total_steps']
