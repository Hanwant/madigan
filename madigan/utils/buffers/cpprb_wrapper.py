import pickle

import cpprb

from .common import DQNTYPES
from ..data import SARSD, State

def make_env_dict_cpprb(agent_class, obs_shape, action_shape):
    if agent_class in DQNTYPES:
        env_dict = {
            "state_price": {
                "shape": obs_shape
            },
            "state_portfolio": {
                "shape": action_shape
            },
            "action": {
                "shape": action_shape
            },
            "next_state_price": {
                "shape": obs_shape
            },
            "next_state_portfolio": {
                "shape": action_shape
            },
            "reward": {},
            "done": {},
            "discounts": {}
        }
    else:
        raise NotImplementedError("cpprb spec has been implemented only for "
                                  "the following agent_classes: "
                                  f"{DQNTYPES}")
    return env_dict

class ReplayBufferC:
    """
    Acts as a base class and a factory,
    Its children are wrappers for an rb from the cpprb library
    cls.from_agent method returns an instance of the following:
    - ReplayBufferC_SARSD - for agents in DQNTYPES I.e only needs sarsd

    """
    def add(self, *kw):
        raise NotImplementedError()

    def sample(self, size):
        raise NotImplementedError()

    def get_all_transitions(self):
        raise NotImplementedError()

    @classmethod
    def from_agent(cls, agent):
        agent_type = type(agent).__name__
        nstep_return = agent.nstep_return
        discount = agent.discount
        obs_shape = agent.input_shape
        env_dict = make_env_dict_cpprb(agent_type, obs_shape,
                                       agent.action_space.shape)
        if agent_type in DQNTYPES:
            return ReplayBufferC_SARSD(
                agent.replay_size,
                env_dict=env_dict,
                Nstep={
                    "size": nstep_return,
                    "gamma": discount,
                    "rew": "reward",
                    "next": ["next_state_price", "next_state_port"]
                })
        raise NotImplementedError("cpprb wrapper has been implemented " +
                                  "only for the following agent_classes: " +
                                  f"{DQNTYPES}")


class ReplayBufferC_SARSD(cpprb.ReplayBuffer, ReplayBufferC):
    """
    Replay Buffer wrapper for cpprb for agents in DQNTYPES.
    Wraps replay buffer from cpprb to maintain consistent api
    Uses SARSD and State dataclasses as intermediaries for i/o
    when sampling and adding
    """
    def __len__(self):
        return self.get_stored_size()

    def sample(self, size):
        data = super().sample(size)
        sarsd = SARSD(
            State(data['state_price'],
                  data['state_portfolio']), data['action'], data['reward'],
            State(data['next_state_price'], data['next_state_portfolio']),
            data['done'])
        return sarsd

    def add(self, sarsd):
        super().add(state_price=sarsd.state.price,
                    state_portfolio=sarsd.state.portfolio,
                    reward=sarsd.reward,
                    next_state_price=sarsd.next_state.price,
                    next_state_portfolio=sarsd.next_state.portfolio,
                    done=sarsd.done)

    def save_to_file(self, savepath):
        """
        Extracts transitions and saves them to file
        Instead of pickling the class - not supported (Cython)
        """
        transitions = self.get_all_transitions()
        with open(savepath, 'wb') as f:
            pickle.dump(transitions, f)

    def load_from_file(self, loadpath):
        if loadpath.is_file():
            with open(loadpath, 'rb') as f:
                transitions = pickle.load(f)
                super().add(transitions)

