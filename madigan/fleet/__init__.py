from .algorithm.dqn import DQN
#from .net.conv_model import ConvNet
#from .net.mlp_model import MLPNet


def make_agent(config):
    if config.agent_type == "DQN":
        return DQN.from_config(config)
    else:
        raise NotImplementedError(f"Agent type {config.agent_type} not implemented")
