from .conv_net import ConvNet, ConvNetAE, ConvNetCurl
from .conv_net_ddpg import ConvCriticQ as ConvCriticQ_DDPG
from .conv_net_ddpg import ConvPolicyDeterministic as ConvPolicy_DDPG
from .conv_net_sac import ConvPolicySACD, TwinQNetwork
from .series_net import SeriesNetQ
from .conv_net_iqn import ConvNetIQN
from .lstm_net import LSTMNet

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
    model_na = NotImplementedError(f"{model_type} not Implemented " +
                                   f"for agent {agent_type}")
    if agent_type in ("DQN", "DQNReverser", "DQNController", "DQNMixedActions"):
        if model_type in ("ConvNet", "ConvNetQ"):
            return ConvNet
        if model_type in ("SeriesNet", "SeriesNetQ"):
            return SeriesNetQ
        raise model_na
    if agent_type in ("DQNRecurrent", ):
        if model_type in ("LSTM", "LSTMNet"):
            return LSTMNet
    if agent_type in ("DQNAE", ):
        if model_type in ("ConvNet", "ConvNetQ", "ConvNetAE"):
            return ConvNetAE
    if agent_type in ("DQNCURL", "DQNReverserCURL"):
        if model_type in ("ConvNet", "ConvNetQ", "ConvNetCURL",
                          "ConvNetCurl"):
            return ConvNetCurl
    if agent_type in ("IQN", "IQNReverser", "IQNController", "IQNMixedActions"):
        if model_type in ("ConvNet", "ConvNetIQN"):
            return ConvNetIQN
        raise model_na
    # if agent_type in ("IQNCURL", ):
    #     if model_type in ("ConvNet", "ConvNetIQN", "ConvNetCurl"):
    #         return ConvNetIQNCURL
    if agent_type in ("DDPG", "DDPGDiscretized"):
        if model_type in ("ConvCriticQ", ):
            return ConvCriticQ_DDPG
        if model_type in ("ConvPolicyDeterministic", ):
            return ConvPolicy_DDPG
        raise model_na
    if agent_type in ("SACDiscrete", ):
        if model_type in ("ConvPolicySACD", ):
            return ConvPolicySACD
        if model_type in ("TwinQNetwork", ):
            return TwinQNetwork

    raise NotImplementedError(f"model {model_type} for agent " +
                              "{agent_type} not Implemented")
