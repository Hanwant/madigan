from .dqn import DQN
from .conv_model import ConvModel
from .mlp_model import MLPModel


def make_agent(config):
    if config.agent_type == "DQN":
        return DQN(config)
    else:
        raise NotImplementedError(f"Agent type {config.agent_type} not implemented")
