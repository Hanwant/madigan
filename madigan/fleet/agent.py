from abc import ABC, abstractmethod

class Agent(ABC):
    """ Base Class for Agents"""
    def __init__(self, name=None):
        self.name = name or ""

    @abstractmethod
    def __call__(self, state):
        pass

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def load_state(self):
        pass

    def __repr__(self):
        return type(self).__name__ + f'id: {self.name}'
