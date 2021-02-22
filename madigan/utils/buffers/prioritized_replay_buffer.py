"""
Basic implementation of priorituzed replay buffer.
Leverages the SegmentTree classes.
"""
import numpy as np
import torch

from .replay_buffer import ReplayBuffer
from .segment_tree import SumTree, MinTree
from ..data import SARSD


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self,
                 size,
                 nstep_return,
                 discount,
                 reward_shaper_config={'reward_shaper': 'sum_default'},
                 alpha=0.6,
                 beta=0.4,
                 beta_steps=10**5,
                 min_pa=0.,
                 max_pa=1.,
                 eps=0.01):
        super().__init__(size, nstep_return, discount, reward_shaper_config)
        self.alpha = alpha
        self.beta = beta
        self.beta_diff = (1. - beta) / beta_steps
        self.min_pa = min_pa
        self.max_pa = max_pa
        self.eps = eps
        tree_size = 1
        while tree_size < size:
            tree_size <<= 1
        self.tree_sum = SumTree(tree_size)
        self.tree_min = MinTree(tree_size)
        self.cached_idxs = None

    @classmethod
    def from_agent(cls, agent):
        return cls(agent.replay_size, agent.nstep_return, agent.discount,
                   agent.reward_shaper_config, agent.per_alpha, agent.per_beta,
                   agent.per_beta_steps)

    @classmethod
    def from_config(cls, config):
        aconf = config.agent_config
        return cls(aconf.replay_size, aconf.nstep_return, aconf.discount,
                   config.reward_shaper_config, aconf.per_alpha,
                   aconf.per_beta, aconf.per_beta_steps)

    def _add_to_replay(self, nstep_sarsd: SARSD):
        super()._add_to_replay(nstep_sarsd)
        self.tree_min[self.current_idx] = self.max_pa
        self.tree_sum[self.current_idx] = self.max_pa

    def sample(self, n: int):
        # assert self.cached_idxs is None, "Update priorities before sampling"

        total_pa = self.tree_sum.reduce(0, self.filled)
        rand = np.random.rand(n) * total_pa
        self.cached_idxs = [self.tree_sum.find_prefixsum_idx(r) for r in rand]
        self.beta = min(1., self.beta + self.beta_diff)

        weight = self._calc_weight(self.cached_idxs)
        batch = self._sample(self.cached_idxs)
        # batch = [self._buffer[idx] for idx in self.cached_idxs]
        return batch, weight

    def _calc_weight(self, idxs: list) -> np.ndarray:
        min_pa = self.tree_min.reduce(0, self.filled)
        weight = [(self.tree_sum[i] / min_pa)**-self.beta for i in idxs]
        weight = np.array(weight, dtype=np.float32)
        return weight

    def update_priority(self, td_error: torch.Tensor):
        assert self.cached_idxs is not None, "Sample another batch before updating priorities"
        assert len(td_error.shape) == 1
        pa = self._calc_pa(td_error).cpu().numpy()
        for i, idx in enumerate(self.cached_idxs):
            self.tree_sum[idx] = pa[i]
            self.tree_min[idx] = pa[i]
        self.cached_idxs = None

    def _calc_pa(self, td_error: torch.Tensor) -> torch.Tensor:
        return torch.clip((td_error + self.eps)**self.alpha, self.min_pa,
                          self.max_pa)
