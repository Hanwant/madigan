from .replay_buffer import ReplayBuffer
from .prioritized_replay_buffer import PrioritizedReplayBuffer
from .common import DQNTYPES


def make_buffer_from_agent(agent):
    """ Factory method to make replay buffers from agent instance """
    if type(agent).__name__ in DQNTYPES:
        if agent.prioritized_replay:
            return PrioritizedReplayBuffer.from_agent(agent)
        return ReplayBuffer.from_agent(agent)
    raise ValueError(f"Only replay buffers for DQNTYPES : {DQNTYPES} "
                        "have been implemented")

def make_buffer_from_config(config):
    """ Factory method to make replay buffers from config """
    aconf = config.agent_config
    if aconf.agent_type in DQNTYPES:
        if aconf.prioritized_replay:
            return PrioritizedReplayBuffer.from_config(config)
        return ReplayBuffer.from_config(config)
    raise ValueError(f"Only replay buffers for DQNTYPES : {DQNTYPES} "
                    "have been implemented")

