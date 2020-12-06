from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch

from ...utils.data import SARSD, State


class Agent(ABC):
    """
    Base class for all agents
    """
    def __init__(self,
                 env,
                 preprocessor,
                 input_shape: tuple,
                 action_space,
                 discount: float,
                 nstep_return: int,
                 savepath=None):
        self._env = env
        self._preprocessor = preprocessor
        self.input_shape = input_shape
        self._action_space = action_space
        self.discount = discount
        self.nstep_return = nstep_return
        self.savepath = Path(savepath)
        self.training_steps = 0
        self.env_steps = 0
        self.branch = 0
        # if not self.savepath.is_dir():
        #     self.savepath.mkdir(parents=True)
        # if (self.savepath/'main.pth').is_file():
        #     self.load_state()

    @property
    def action_space(self):
        return self._action_space

    @property
    def env(self):
        return self._env

    @property
    def preprocessor(self):
        return self._preprocessor

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        pass

    @abstractmethod
    def get_action(self, state: State) -> np.ndarray:
        """
        Fundamental function.
        Returns vector of units to purchase (+/-) for each asset
        """
        pass

    @abstractmethod
    def train_step(self, sarsd=None) -> dict:
        """
        May use provided sarsd or may sample from own internal buffer
        to perform a batched training step
        Returns a dict of metrics I.e {'loss': loss, 'td_error': td_error}
        """
        pass

    @abstractmethod
    def step(self, n: int) -> List[dict]:
        """
        Generator which must be initialized by calling loop = iter(agent.step(n)).
        Fundamental loop in which the agent accumulates experiences via env interaction
        and trains at the designated interval.
        Performs n training steps
        and yields a list of training metrics at the interval specified by config
        """
        pass

    @abstractmethod
    def explore(self, state):
        """
        For interacting with the environment in training phase
        I.e may implement an eps-greedy exploration policy
        """
        pass

    @abstractmethod
    def reset_state(self) -> State:
        """
        Implements state resetting
        Most likely will call a generic function
        by passing self, env, and preprocessor
        """

    @abstractmethod
    def test_episode(self, test_steps=None, reset=False):
        """
        To test agent performance, the agent interacts with the env
        until either test_steps has been reached or done is returned
        """

    def __call__(self, *args, **kw):
        return self.get_action(*args, **kw)

    @abstractmethod
    def load_state(self, branch: str = "main"):
        """
        Loads models and state from self.savepath/{branch}.pth
        branch: str = branch name
        state has to include:
            - env_steps
            - training_steps
        optional:
            - model state
            - eps
            - lr
            - etc
        """

    @abstractmethod
    def save_state(self, branch: str = "main"):
        """
        Saves models and state into self.savepath/{branch}.pth
        Overwrites it exists
        branch: str = branch name
        state has to include:
            - env_steps
            - training_steps
        optional:
            - model state
            - eps
            - lr
            - etc
        """

    def checkpoint(self):
        check = self.savepath / f'checkpoint_{self.training_steps}'
        self.save_state(check)


def test_episode(agent: Agent,
                 env,
                 preprocessor,
                 test_steps: int,
                 reset: bool = True) -> dict:
    """
    Wrapper function for testing Agents out of sample
    Allows decoupling of agent from it's internal env which was used to train
    so as to easily allow testing the agent with different env and preprocessor
    I.e a test/val dataset
    @params:
        agent: OffPolicyQ
        env : Env
        preprocessor:
        test_steps: int = number of env interaction steps to perform
        reset: bool = whether or not to call env.reset() before starting
    """
    prev_env = agent._env
    prev_preprocessor = agent._preprocessor
    agent._env = env
    agent._preprocessor = preprocessor
    test_metrics = agent.test_episode(test_steps, reset=reset)
    agent._env = prev_env
    agent._preprocessor = prev_preprocessor
    return test_metrics
