from .net.conv_net import ConvNet, ConvCriticQ, ConvPolicyDeterministic
from .net.series_net import SeriesNetQ
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
    model_na = NotImplementedError(f"{model_type} not Implemented" +
                                   f"for agent {agent_type}")
    if agent_type in ("DQN", "DQNReverser"):
        if model_type in ("ConvNet", "ConvNetQ"):
            return ConvNet
        if model_type in ("SeriesNet", "SeriesNetQ"):
            return SeriesNetQ
        raise model_na
    if agent_type in ("IQN", ):
        if model_type in ("ConvNet", "ConvNetIQN"):
            return ConvNetIQN
        raise model_na
    if agent_type in ("DDPG", "DDPGDiscretized"):
        if model_type == "ConvCriticQ":
            return ConvCriticQ
        if model_type == "ConvPolicyDeterministic":
            return ConvPolicyDeterministic
        raise model_na

    raise NotImplementedError(f"model {model_type} for agent " +
                              "{agent_type} not Implemented")
