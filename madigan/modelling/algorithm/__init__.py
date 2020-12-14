from .dqn import DQN, DQNReverser, DQNReverserCURL
from .iqn import IQN, IQNReverser
from .ddpg import DDPG, DDPGDiscretized
# from .net.mlp_model import MLPNet


def make_agent(config):
    if config.agent_type in ("DQN", "DQNReverser", "DQNReverserCURL", "IQN",
                             "IQNReverser", "DDPG", "DDPGDiscretized"):
        return globals()[config.agent_type].from_config(config)
    raise NotImplementedError(f"Agent type {config.agent_type} not implemented")
