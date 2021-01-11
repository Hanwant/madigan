from functools import partial
import math
import numpy as np
import numba as nb
from ..data import SARSD

import warnings


def sum_default(nstep_buffer, discounts):
    """ Default return aggregation """
    return sum([
        dat.reward * discount for dat, discount in zip(nstep_buffer, discounts)
    ])


global print_i
print_i = 0


def cosine_similarity(p: np.ndarray, q: np.ndarray):
    ppp = p
    qqq = q
    norm_p = np.sqrt((p**2).sum(-1))
    norm_q = np.sqrt((q**2).sum(-1))
    return ((p * q).sum(-1) / (norm_p * norm_q)).mean()


def cosine_port_shaper(nstep_buffer, discounts, desired_portfolio,
                       cosine_temp):
    """
    rewards next_state.portfolio depending on it's distance from
    a desired portfolio, measured using cosine similarity.
    must be tuned to prevent exploiting rewards by getting to target
    port and staying there. - soft actor critic would help
    """
    rewards = sum([
        discount *
        (dat.reward + cosine_temp *
         cosine_similarity(dat.next_state.portfolio[-1], desired_portfolio))
        for dat, discount in zip(nstep_buffer, discounts)
    ])
    global print_i
    if print_i > 100:
        cosine_reward = sum([
            cosine_similarity(dat.next_state.portfolio, desired_portfolio)
            for dat in nstep_buffer
        ])
        normal_reward = sum([dat.reward for dat in nstep_buffer])
        print(normal_reward, cosine_reward, cosine_reward * cosine_temp)
        print_i = 0
    print_i += 1
    return rewards


def sharpe_shaper(nstep_buffer, discounts, benchmark=0.):
    """ Sharpe return aggregation """
    if len(nstep_buffer) == 1:  # Heuristic for if there is only 1 value
        benchmark = benchmark if isinstance(benchmark, float) else benchmark[0]
        diff = nstep_buffer[0].reward - benchmark
        if diff == 0.:  # prevent div by 0. in denom
            return 0.
        return diff / math.sqrt(diff**2)
    if not isinstance(benchmark, float):  # if benchmark is an array
        diffs = np.array([
            (dat.reward - ref) * discount
            for dat, ref, discount in zip(nstep_buffer, benchmark, discounts)
        ])
    else:  # if benchmark is a constant scalar - default
        diffs = np.array([(dat.reward - benchmark) * discount
                          for dat, discount in zip(nstep_buffer, discounts)])
    num = diffs.mean()
    denom = math.sqrt((diffs**2).sum() / (len(diffs) - 1))
    if denom == 0.:  # prevent div by 0.
        return 0.
    global print_i
    print_i += 1
    if print_i > 100:
        print(num, denom, num / denom)#, .1 * (num / denom))  # *.1 for scaling
        print_i = 0
    return np.clip((num / denom), -1., 1.)


def sortino_shaperA(nstep_buffer, discounts, benchmark=0.):
    """ Sortino return aggregation """
    if len(nstep_buffer) == 1:  # Heuristic for if there is only 1 value
        benchmark = benchmark if isinstance(benchmark, float) else benchmark[0]
        diff = nstep_buffer[0].reward - benchmark
        if diff == 0.:  # prevent div by 0. in denom
            return 0.
        if diff < 0:
            return np.clip(diff / math.sqrt(diff**2), -1., 1.)
        return 1.
    if not isinstance(benchmark, float):  # if benchmark is an array
        diffs = np.array([
            (dat.reward - ref) * discount
            for dat, ref, discount in zip(nstep_buffer, benchmark, discounts)
        ])
    else:  # if benchmark is a constant scalar - default
        diffs = np.array([(dat.reward - benchmark) * discount
                          for dat, discount in zip(nstep_buffer, discounts)])
    num = diffs.mean()
    downside = diffs[np.where(diffs < 0)[0]]
    if len(downside) == 0:
        if num == 0:
            return 0.
        # print('all upside')
        return 1.
    denom = math.sqrt((downside**2).sum() / (len(diffs) - 1))
    global print_i
    print_i += 1
    if print_i > 100:
        print(num, denom, num / denom, .1 * (num / denom))
        print_i = 0
    return np.clip(.1 * (num / denom), -1., 1.)


def sortino_shaperB(nstep_buffer, discounts, benchmark=0.):
    """
    Another version - kinda sortino style. Instead of adjusting for downside
    volatility, returns are squared in magnitude if they are negative.

    """
    if len(nstep_buffer) == 1:  # Heuristic for if there is only 1 value
        benchmark = benchmark if isinstance(benchmark, float) else benchmark[0]
        diff = nstep_buffer[0].reward - benchmark
        if diff == 0.:  # prevent div by 0. in denom
            return 0.
        if diff < 0:
            if diff < -1.:
                return -1.
            return -((-diff)**(1 / 2))
        return diff
    if not isinstance(benchmark, float):  # if benchmark is an array
        diffs = np.array([
            (dat.reward - ref) * discount
            for dat, ref, discount in zip(nstep_buffer, benchmark, discounts)
        ])
    else:  # if benchmark is a constant scalar - default
        diffs = np.array([(dat.reward - benchmark) * discount
                          for dat, discount in zip(nstep_buffer, discounts)])
    raw_sum = diffs.sum()
    diffs = np.where(diffs < -1., -1., diffs)
    downside_idx = np.where(diffs < 0.)[0]
    diffs[downside_idx] = -((-diffs[downside_idx])**(2 / 3))
    global print_i
    print_i += 1
    if print_i > 100:
        print(raw_sum, diffs.sum())
        print_i = 0
    return diffs.sum()


class NStepBuffer:
    """
    Utility class to prevent code duplicaiton.
    It must however be co-ordinated by a class using it (I.e ReplayBuffer)
    It doesn't have access to the main replay buffer and doesn't care if the
    numbers of sampels in its buffer is > nstep. It is the containers
    responsibility to flush and add the processesd samples to a main buffer.
    """
    def __init__(self, nstep, discount, reward_shaper_config):
        self.nstep = nstep
        self.discount = discount
        self.discounts = [math.pow(self.discount, i) for i in range(nstep)]
        self._buffer = []
        # self.aggregate_rewards = partial(sortino_aggregate, self._buffer,
        #                                  self.discounts)
        self.aggregate_rewards = self.make_reward_shaper(reward_shaper_config)

    def make_reward_shaper(self, shaper_config):
        """
        Constructs rewards shapers by binding self._buffer and self.discount
        to the shaper function using partial - keeps reference to both.
        """
        shaper_type = shaper_config['reward_shaper']
        if shaper_type in ("cosine_similarity", "cosine_port_shaper"):
            desired_portfolio = np.array(shaper_config['desired_portfolio'])
            temp = shaper_config['cosine_temp']
            return partial(cosine_port_shaper, self._buffer, self.discounts,
                           desired_portfolio, temp)
        if shaper_type in globals():
            print('using ', shaper_type)
            return partial(globals()[shaper_type], self._buffer,
                           self.discounts)
        if shaper_type in ('None', None, 'none'):
            return partial(sum_default, self._buffer, self.discounts)
        raise NotImplementedError(
            f"Reward Shaper type {shaper_type} not implemented")

    def add(self, sarsd: SARSD) -> None:
        self._buffer.append(sarsd)

    def full(self) -> bool:
        return len(self._buffer) >= self.nstep

    def pop_nstep_sarsd(self) -> SARSD:
        """
        Calculates nstep discounted return from the nstep buffer
        and returns the sarsd with the adjusted return and next_state offset to t+n
        """
        # reward = sum([
        #     self.discounts[i] * dat.reward
        #     for i, dat in enumerate(self._buffer)
        # ])
        reward = self.aggregate_rewards()
        nstep_sarsd = self._buffer.pop(0)
        nstep_sarsd.reward = reward
        if len(self._buffer) > 0:
            # nstep_idx = min(self.nstep, len(self)) - 1
            nstep_idx = -1
            nstep_sarsd.next_state = self._buffer[nstep_idx].next_state
            nstep_sarsd.done = self._buffer[nstep_idx].done
            # if self._nstep_buffer[-1].done:
            #     nstep_sarsd.done = 1
        return nstep_sarsd

    def flush_nstep_buffer(self):
        """
        Useful to call at end of episodes (I.e if not done.)
        """
        out = []
        while self.full():
            out.append(self.pop_nstep_sarsd())
        return out

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)
