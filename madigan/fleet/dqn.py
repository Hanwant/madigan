from pathlib import Path
import torch
import torch.nn.functional as F
from .agent import Agent
from .conv_model import ConvModel
from .mlp_model import MLPModel
from ..utils import get_model_class

# p = type('params', (object, ), params)

class DQN(Agent):
    """
    Implements a base DQN agent from/to which extensions can inherit/extend
    The instance obj can be called directly to get an action based on state:
        action = dqn(state)
    The method for training is self.train_step(sarsd) where sarsd is a class with array members (I.e (bs, time, feats))
    """
    def __init__(self, config, name=None, device=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        agent_config = config['agent_config']
        m_config = agent_config['model_config']
        self.savepath = agent_config['savepath']

        if Path(self.savepath).is_file():
            config = torch.load(self.savepath)
            self.model_class = get_model_class(m_config['model_class'])
        else:
            self.model_class = get_model_class(m_config['model_class'])

        # import ipdb; ipdb.set_trace()
        self.model_b = self.model_class(**m_config).to(device)
        self.model_t = self.model_class(**m_config).to(device)

        if Path(self.savepath).is_file():
            self.load_state()
        else:
            # self.save_state()
            self.training_steps = 0
            self.total_steps = 0
        super().__init__(name)

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

    def train_step(self, sarsd):
        pass

    def __call__(self, state, target=True, raw_qvals=False, max_qvals=False):
        with torch.no_grad():
            qvals = self.get_qvals(state, target=target)
            if raw_qvals:
                return qvals
            # max_qvals = qvals.max(-1)[0]
            actions = qvals.max(-1)[1]
        return actions

    def get_qvals(self, state, target=True, device=None):
        device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.tensor(state, dtype=torch.float32)
        if target:
            return self.model_t(state.to(device))
        return self.model_b(state.to(device))
