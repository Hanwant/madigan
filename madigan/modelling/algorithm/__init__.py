from .dqn import DQN, DQNCURL, DQNAE
from .dqn_controller import DQNController
from .dqn_recurrent import DQNRecurrent
from .iqn import IQN, IQNCURL
from .iqn_controller import IQNController
from .ddpg import DDPG, DDPGDiscretized
from .sac import SACDiscrete
# from .net.mlp_model import MLPNet


def make_agent(config):
    if config.agent_type in globals():
        return globals()[config.agent_type].from_config(config)
    raise NotImplementedError(
        f"Agent type {config.agent_type} not implemented")
