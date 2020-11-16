from .algorithm.dqn import DQN, DQNReverser
from .algorithm.iqn import IQN
from .algorithm.ddpg import DDPG, DDPGDiscretized
# from .net.mlp_model import MLPNet


def make_agent(config):
    if config.agent_type in ("DQN", "DQNReverser", "IQN", "DDPG",
                             "IQN", "DDPG", "DDPGDiscretized"):
        return globals()[config.agent_type].from_config(config)
    # if config.agent_type == "DQN":
    #     return DQN.from_config(config)
    # if config.agent_type == "DQNReverser":
    #     return DQNReverser.from_config(config)
    # if config.agent_type == "IQN":
    #     return IQN.from_config(config)
    # if config.agent_type == "DDPG":
    #     return DDPG.from_config(config)
    # if config.agent_type == "DDPGDiscretized":
    #     return DDPGDiscretized.from_config(config)

    else:
        raise NotImplementedError(f"Agent type {config.agent_type} not implemented")

