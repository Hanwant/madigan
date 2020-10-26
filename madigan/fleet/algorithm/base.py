from abc import ABC, abstractmethod
import numpy as np
from ...utils import ReplayBuffer

class Agent(ABC):
    """
    Base class for all agents
    """
    def __init__(self, env, preprocessor, input_space, action_space, discount,
                 savepath=None):
        self._env = env
        self._preprocessor = preprocessor
        self.input_space = input_space
        self.action_space = action_space
        self.discount = discount
        self.savepath = savepath
        self.training_steps = 0
        self.env_steps = 0
        if not self.savepath.is_dir():
            self.savepath.mkdir(parents=True)
        if (self.savepath/'main.pth').isfile():
            self.load_state()

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        pass

    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def __call__(self, *args, **kw):
        return self.get_action(*args, **kw)

    @abstractmethod
    def load_state(self):
        pass

    @abstractmethod
    def save_state(self):
        pass



class OffPolicyQ(Agent):
    """
    Base class for all off policy agents with experience replay buffers
    """
    def __init__(self, env, preprocessor, input_space, action_space, discount,
                 nstep_return, replay_size, replay_min_size, savepath):
        super().__init__(env, preprocessor, input_space, action_space, discount,
                         savepath)
        self.nstep_return = nstep_return
        self.buffer = ReplayBuffer(replay_size, nstep_return, discount)
        self.replay_min_size = replay_min_size
        self.action_atoms = self.action_space[-1]
        if len(self.action_space) > 1:
            self.nassets = action_space[0]
        else:
            self.nassets = 1
        self.centered_actions = np.arange(self.action_atoms) - self.action_atoms // 2


