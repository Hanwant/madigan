# from typing import Union
# from ...utils.config import Config

# def make_model(config: Union[dict, Config], model_type):
#     """
#     agent_type refers to an exact class name
#     model_type: choose from ('critic', 'actor')
#     """
#     config = Config(config)
#     assert model_type in ('critic', 'actor')
#     model_na = NotImplementedError(f"{model_class} not Implemented" +\
#                                    f"for agent {agent_type}")
#     agent_class = config.agent_class
#     model_class = config.model_class
#     if agent_class in ("DQN", ):
#         if model_class in ("ConvNet", ):
#             return ConvNet
#         raise model_na
#     if agent_class in ("IQN", ):
#         if model_class in ("ConvNet", "ConvNetIQN"):
#             return ConvNetIQN
#         raise model_na
#     if agent_class in ("DDPG", ):
#         if model_class == "ConvCriticQ":
#             return ConvCriticQ
#         if model_class == "ConvPolicyDeterministic":
#             return ConvPolicyDeterministic
#         raise model_na

#     raise NotImplementedError(f"models for agent {agent_type} not Implemented")
