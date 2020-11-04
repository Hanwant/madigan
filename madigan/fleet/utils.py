from .net.conv_net import ConvNet
from .net.conv_net_iqn import ConvNetIQN

def get_model_class(agent_type, model_type):
    """
    agent_type refers to an exact class name
    whereas model_type may be a subname
    I.e agent_type = DQN
        model_type = ConvNet
    returns ConvNet
    OR
        agent_type = IQN
        model_type = ConvNet
    returns ConvNetIQN
    """
    if agent_type in ("DQN", ):
        if model_type in ("ConvNet", ):
            return ConvNet
        raise NotImplementedError(f"model of type {model_type} not Implemented")
    if agent_type in ("IQN", ):
        if model_type in ("ConvNet", "ConvNetIQN"):
            return ConvNetIQN
        raise NotImplementedError(f"model of type {model_type} not Implemented")
    raise NotImplementedError(f"models for agent {agent_type} not Implemented")
