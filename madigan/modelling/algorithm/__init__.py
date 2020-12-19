from .dqn import DQN, DQNCURL
from .iqn import IQN, IQNCURL
from .ddpg import DDPG, DDPGDiscretized
from .sac import SACDiscrete
# from .net.mlp_model import MLPNet


def make_agent(config):
    if config.agent_type in ("DQN", "DQNCURL", "IQN", "IQNCURL",
                             "DDPG", "DDPGDiscretized",
                             "SACDiscrete"):
        return globals()[config.agent_type].from_config(config)
    raise NotImplementedError(f"Agent type {config.agent_type} not implemented")
