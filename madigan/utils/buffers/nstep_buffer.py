"""
As reward aggregation is done in the nstep buffer, this module contains
the different methods for reward aggregation/shaping including:
DSR
DDR
sharpe_shaper (naive)
sortino_shaperA (naive)
sortino_shaperB (naive)  - greater contribution from negative rewards
cosine_port_shaper - cosine similarity between desired portfolio and portfolio
                     resulting from action

"""
from functools import partial
import math
import numpy as np
import numba as nb
from ..data import SARSD

import warnings

EPS = np.finfo(np.float32).eps


def sum_default(nstep_buffer, discounts):
    """ Default return aggregation """
    return sum([
        dat.reward * discount for dat, discount in zip(nstep_buffer, discounts)
    ])


class _DSR:
    """Differential Sharpe Ratio - Moody and Saffell
    Currently treats discounted aggregation of rewards indepdnently
    - doesn't update self.A and self.B for successive rewards
    """
    def __init__(self, adaptation_rate, nstep_buffer, discounts):
        self.n = adaptation_rate
        self.nstep_buffer = nstep_buffer
        self.discounts = np.array(discounts)
        self.A = None
        self.B = None
        self.i = 0

    def __call__(self):
        """ Called JUST ONCE to determine shape of rewards.
        Afterwards this implementation is replaced by __main_func__
        By changing class instance to DSR
        This was done because __call__ is bound to the class and so
        could not be monkey patched as normal - without affecting all other
        instances.
        """
        reward = self.nstep_buffer[0].reward
        if isinstance(reward, float):
            self.A = 0.
            self.B = 0.
        elif isinstance(reward, np.ndarray):
            self.A = np.zeros(reward.shape[0])
            self.B = np.zeros(reward.shape[0])
            self.discounts = self.discounts[:, None]
        self.__class__ = DSR
        return self.__main_func__()

    def __main_func__(self):
        # rewards = sum([
        #     discount * self.calculate_dsr(dat.reward)
        #     for dat, discount in zip(self.nstep_buffer, self.discounts)
        # ]) / len(self.nstep_buffer)
        rewards = np.array([dat.reward for dat in self.nstep_buffer])
        rewards = (self.discounts[:len(rewards)] *
                   self.calculate_dsr(rewards)).sum(0) / len(rewards)
        # as this will be called when popping from the nstep buffer
        # this is an appropriate place to update params
        self.update_parameters(self.nstep_buffer[0].reward)
        self.i += 1
        if self.i % 100 == 0:
            print(rewards, rewards / len(self.nstep_buffer))
        return np.clip(rewards, -1., 1.)  # clip to (-1, 1)

    def calculate_dsr(self, raw_return):
        dA = raw_return - self.A
        dB = raw_return**2 - self.B
        dsr = (self.B * dA -
               (self.A * dB) / 2) / (((self.B - self.A**2)**2)**(3 / 4) + EPS)
        return dsr

    def update_parameters(self, raw_return):
        dA = raw_return - self.A
        dB = raw_return**2 - self.B
        self.A += self.n * dA
        self.B += self.n * dB


class DSR(_DSR):
    """ After the first time a _DSR instance is called,
    This class gets patched in."""
    def __call__(self):
        return self.__main_func__()


class _DDR:
    """Differential Downside Ratio (sortino) - Moody and Saffell """
    def __init__(self, adaptation_rate, nstep_buffer, discounts):
        self.n = adaptation_rate
        self.nstep_buffer = nstep_buffer
        self.discounts = np.array(discounts)
        self.A = None
        self.B = None
        self.i = 0

    def __call__(self):
        """ Called JUST ONCE to determine shape of rewards.
        Afterwards this implementation is replaced by __main_func__
        By Patching in DDR class to this instance
        """
        reward = self.nstep_buffer[0].reward
        if isinstance(reward, float):
            self.A = 0.
            self.B = 0.
        elif isinstance(reward, np.ndarray):
            self.A = np.zeros(reward.shape[0])
            self.B = np.zeros(reward.shape[0])
            self.discounts = self.discounts[:, None]
            print(self.discounts.shape)
        self.__class__ = DDR
        return self.__main_func__()

    def __main_func__(self):
        # rewards = sum([
        #     discount * self.calculate_ddr(dat.reward)
        #     for dat, discount in zip(self.nstep_buffer, self.discounts)
        # ]) / len(self.nstep_buffer)
        rewards = np.array([dat.reward for dat in self.nstep_buffer])
        rewards = (self.discounts[:len(rewards)] *
                   self.calculate_ddr(rewards)).sum(0) / len(rewards)
        # as this will be called when popping from the nstep buffer
        # this is an appropriate place to update params
        self.update_parameters(self.nstep_buffer[0].reward)
        self.i += 1
        if self.i % 100 == 0:
            print(rewards)
        return np.clip(rewards, -1., 1)  # clip to (-1, 1)

    def calculate_ddr(self, raw_return):
        # dA = raw_return - self.A
        # dB = min(raw_return, 0)**2 - self.B
        # ddr = (self.B * dA -
        #        (self.A * dB) / 2) / (((self.B - self.A**2)**2)**(3 / 4) + EPS)
        ddr = np.where(raw_return > 0.,
                       (raw_return - self.A / 2) / (np.sqrt(self.B) + EPS),
                       (self.B * (raw_return - self.A / 2) -
                        (self.A * raw_return**2) / 2) /
                       (self.B**(3 / 2) + EPS))
        return ddr

    def update_parameters(self, raw_return):
        dA = raw_return - self.A
        dB = np.minimum(raw_return, 0.)**2 - self.B
        self.A += self.n * dA
        self.B += self.n * dB


class DDR(_DDR):
    def __call__(self):
        return self.__main_func__()


global PRINT_I
PRINT_I = 0


def cosine_similarity(p: np.ndarray, q: np.ndarray):
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
    global PRINT_I
    if PRINT_I > 100:
        cosine_reward = sum([
            cosine_similarity(dat.next_state.portfolio, desired_portfolio)
            for dat in nstep_buffer
        ])
        normal_reward = sum([dat.reward for dat in nstep_buffer])
        print(normal_reward, cosine_reward, cosine_reward * cosine_temp)
        PRINT_I = 0
    PRINT_I += 1
    return rewards


def sharpe_shaper(nstep_buffer, discounts, benchmark=0.):
    """
    Sharpe return aggregation. Use numpy ufuncs to generalize to
    reward vectors.
    """
    if len(nstep_buffer) == 1:  # Heuristic for if there is only 1 value
        benchmark = benchmark if isinstance(benchmark, float) else benchmark[0]
        diff = nstep_buffer[0].reward - benchmark
        diff = np.where(diff != 0., diff, 0.)
        return (diff / np.sqrt(diff**2))
    if not isinstance(benchmark, float):  # if benchmark is an array
        diffs = np.array([
            (dat.reward - ref) * discount
            for dat, ref, discount in zip(nstep_buffer, benchmark, discounts)
        ])
    else:  # if benchmark is a constant scalar - default
        diffs = np.array([(dat.reward - benchmark) * discount
                          for dat, discount in zip(nstep_buffer, discounts)])
    num = diffs.mean(0)
    denom = np.sqrt((diffs**2).sum(0) / (len(diffs) - 1))
    # if denom == 0.:  # prevent div by 0.
    #     return 0.
    global PRINT_I
    PRINT_I += 1
    if PRINT_I > 100:
        print(num, denom,
              .1 * num / denom)  #, .1 * (num / denom))  # *.1 for scaling
        PRINT_I = 0
    # returns num/denom where denom != 0. else fills that entry with 0.
    # potentially expensive as it creates zeros vector each time!
    # make one big global one and then index into it?
    out = np.divide(num, denom, where=denom != 0, out=np.zeros_like(denom))
    return np.clip(.1 * out, -1., 1.)


def sortino_shaperA(nstep_buffer, discounts, benchmark=0., exp=2):
    """ Sortino return aggregation """
    if len(nstep_buffer) == 1:  # Heuristic for if there is only 1 value
        benchmark = benchmark if isinstance(benchmark, float) else benchmark[0]
        diff = nstep_buffer[0].reward - benchmark
        downside = (np.abs(diff)**exp)**(1 / exp)
        return np.clip(0.1 * np.where(diff != 0., diff / downside, 0.), -1.,
                       1.)

    if not isinstance(benchmark, float):  # if benchmark is an array
        diffs = np.array([
            (dat.reward - ref) * discount
            for dat, ref, discount in zip(nstep_buffer, benchmark, discounts)
        ])
    else:  # if benchmark is a constant scalar - default
        diffs = np.array([(dat.reward - benchmark) * discount
                          for dat, discount in zip(nstep_buffer, discounts)])

    num = diffs.mean(0)
    downside = np.clip(np.minimum(diffs, 0.), -1., None)
    denom = ((np.abs(downside)**exp) / (len(diffs) - 1))**(1 / exp)
    denom = denom.sum(0)
    denom_zero_case = np.where(num == 0., 0., 1.)
    normal_case = np.clip(.1 * (num / denom), -1., 1.)
    out = np.where(denom != 0., normal_case, denom_zero_case)
    global PRINT_I
    PRINT_I += 1
    if PRINT_I > 100:
        print(num, denom, 10. * out, out)
        PRINT_I = 0
    return out


@np.errstate(invalid='ignore')
def sortino_shaperB(nstep_buffer, discounts, benchmark=0., exp=2):
    """
    Another version - kinda sortino style. Instead of adjusting for downside
    volatility, returns are scaled in magnitude if they are negative.

    Scaled by taking downside ** (1/exp). Because 'downside' is the return
    clipped to be -1. < x < 0, taking the root (1/exp) results in increasing
    the value as if the power was taken (i.e if abs(x) > 1.).

    """
    if len(nstep_buffer) == 1:  # Heuristic for if there is only 1 value
        benchmark = benchmark if isinstance(benchmark, float) else benchmark[0]
        diff = np.clip(nstep_buffer[0].reward - benchmark, -1., None)
        diff = np.where(diff < 0., -(-diff)**(1 / exp), diff)  # scale neg rew
        # diff = np.where(diff == 0., -.01, diff)
        return np.clip(diff, -1., 1.)

    if not isinstance(benchmark, float):  # if benchmark is an array
        diffs = np.array([
            (dat.reward - ref) * discount
            for dat, ref, discount in zip(nstep_buffer, benchmark, discounts)
        ])
    else:  # if benchmark is a constant scalar - default
        diffs = np.array([(dat.reward - benchmark) * discount
                          for dat, discount in zip(nstep_buffer, discounts)])

    raw_sum = diffs.sum(0)
    diffs = np.clip(diffs, -1., None)  # max negative reward = -1.
    # downside_idx = np.where(diffs < 0.)[0]
    # diffs[downside_idx] = -((-diffs[downside_idx])**exp)
    diffs = np.where(diffs < 0., -(-diffs)**(1 / exp), diffs)  # scale neg rew
    global PRINT_I
    PRINT_I += 1
    if PRINT_I > 100:
        print(raw_sum, diffs.sum(0))
        PRINT_I = 0
    return np.clip(diffs.sum(0), -1., 1.)


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

    def make_reward_shaper(self, shaper_config):
        """
        Constructs rewards shapers by binding self._buffer and self.discount
        to the shaper function using partial - keeps reference to both.
        """
        shaper_type = shaper_config['reward_shaper']
        print('using ', shaper_type, 'reward shaper')
        if shaper_type in ("cosine", "cosine_similarity",
                           "cosine_port_shaper"):
            desired_portfolio = np.array(shaper_config['desired_portfolio'])
            temp = shaper_config['cosine_temp']
            return partial(cosine_port_shaper, self._buffer, self.discounts,
                           desired_portfolio, temp)
        if shaper_type in ("sortino_shaperA", "sortino_shaperB"):
            exp = shaper_config["sortino_exp"]
            sortino_shaper = globals()[shaper_type]
            return partial(sortino_shaper,
                           self._buffer,
                           self.discounts,
                           exp=exp)
        if shaper_type in ("DSR", "DDR"):
            adaptation_rate = shaper_config["adaptation_rate"]
            return globals()["_" + shaper_type](adaptation_rate, self._buffer,
                                                self.discounts)
        if shaper_type in globals():
            return partial(globals()[shaper_type], self._buffer,
                           self.discounts)
        if shaper_type in ('None', None, 'none'):
            return partial(sum_default, self._buffer, self.discounts)
        raise NotImplementedError(
            f"Reward Shaper type {shaper_type} not implemented")
