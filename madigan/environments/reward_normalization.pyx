"""
Contains classes for transforming raw rewards (assumed to be log returns).
transformations include:
- sharpe ratio
- sortino ratio
"""
import math
from libc.math cimport sqrt

# from collections import deque
from libcpp.queue cimport queue


def make_reward_normalizer(config):
    conf = config['reward_shaper_config']
    name = conf['reward_shaper']
    if name in ("None", "none", None):
        return NullShaper()
    if name in globals():
        return globals()[name].from_config(conf)
    raise NotImplementedError(f"reward_shaper {name} has not "
                              "been implemented")


cdef class RewardShaper:
    """
    May keep a running estimate of metrics such as mean/var/std.
    """

    @classmethod
    def from_config(cls, config):
        """
        Most likely entry point for initialization
        """
        raise NotImplementedError

    cpdef void reset(self):
        """
        reset running estimates
        """
        raise NotImplementedError

    cpdef double stream(self, double reward):
        """
        Streams in new reward and returns a transformation
        """
        raise NotImplementedError


cdef class NullShaper(RewardShaper):
    """ To maintain interface """
    @classmethod
    def from_config(cls, config):
        return cls()
    def reset(self):
        pass
    def stream(self, reward):
        return reward


cdef class SharpeFixedWindow(RewardShaper):
    """
    Streams in rewards and returns 'sharpe ratio' values based on a rolling
    window estimate of standard deviation

    If sequential rewards were an array, this would be equivalent to:
        rewards / pd.Series(rewards).rolling(window, min_periods=2).std(ddof=0)

    NOTE THIS IS NOT STRICTLY EQUIVALENT TO THE NORMAL SHARPE RATIO as for
    computaiton of the std, a rolling estimate of the mean is used,
    instead of a risk free or target rate of return.
    """
    cdef unsigned int window  #, count
    cdef queue[double] buffer
    cdef double mean_est, ssq

    def __init__(self, window: int):
        self.window = window
        self.buffer = queue[double]()
        self.reset()

    @classmethod
    def from_config(cls, config):
        window = config['window']
        return cls(window)

    cpdef void reset(self):
        """ Needs to be called when environment resets / episode ends. """
        while not self.buffer.empty():
            self.buffer.pop()
        self.mean_est = 0.
        # self.count = 0
        self.ssq = 0.

    cpdef double stream(self, double reward):
        """ Adjusts internal metric estimates and returns std estimate """
        if self.buffer.size() == self.window:
            self.tail_adjust()
        self.head_add(reward)  # done after tail adjust so end doesn't get lost
        if self.buffer.size() <= 1:
            return 0.
        return reward / sqrt((self.ssq + 1e-8)
                                  / self.buffer.size())

    cdef void head_add(self, double value):
        # self.count += 1
        self.buffer.push(value)
        delt = value - self.mean_est
        self.mean_est += delt / self.buffer.size()
        self.ssq += delt * (value - self.mean_est)

    cdef void tail_adjust(self):
        # self.count -= 1
        remove = self.buffer.front()
        self.buffer.pop()
        delt = remove - self.mean_est
        self.mean_est -= delt / self.buffer.size()
        self.ssq -= (delt * (remove-self.mean_est))


cdef class SortinoFixedWindowA(SharpeFixedWindow):
    """
    Streams in rewards and returns a sortino style transformation of reward.
    Rewards are scaled by a rolling window estimate of reward std. If negative,
    rewards are additionally squared in magnitude - hence sortino style.
    This provides extra feedback for negative rewards.

    If sequential rewards were an array, this would be equivalent to:
        std = pd.Series(rewards).rolling(window, min_periods=2).std(ddof=0)
        rewards = rewards if rewards > 0 else -(rewards**2)
        return rewards / std

    """

    cpdef double stream(self, double reward):
        """ Adjusts internal metric estimates and returns std estimate """
        if self.buffer.size() == self.window:
            self.tail_adjust()
        self.head_add(reward)  # done after tail adjust so end doesn't get lost
        if self.buffer.size() <= 1:
            return 0.
        reward = reward / sqrt((self.ssq+1e-8) / self.buffer.size())
        if reward < 0:
            return -1 * (reward * reward)
        else:
            return reward


cdef class SortinoFixedWindowB:
    """
    See SortinofixedwindowA.
    The only difference from A is that the estimate of std here also uses a
    sortino style aggregation, where only negative values contribute to the std.

    """
    cdef unsigned int window , count
    cdef queue[double] buffer
    cdef double mean_est, ssq

    def __init__(self, window: int):
        self.window = window
        self.buffer = queue[double]()
        self.reset()

    @classmethod
    def from_config(cls, config):
        window = config['window']
        return cls(window)

    cpdef void reset(self):
        """ Needs to be called when environment resets / episode ends. """
        while not self.buffer.empty():
            self.buffer.pop()
        self.mean_est = 0.
        self.count = 0
        self.ssq = 0.

    cpdef double stream(self, double reward):
        """ Adjusts internal metric estimates and returns std estimate """
        self.update(reward)  # done after tail adjust so end doesn't get lost
        if self.buffer.size() <= 1:
            return 0.
        reward = reward / sqrt((self.ssq+1e-8) / self.count)
        if reward < 0:
            return -1 * (reward * reward)
        else:
            return reward

    cdef double update(self, double value):
        cdef double delt = value - self.mean_est
        if delt < 0:
            self.count += 1
            self.mean_est += delt / self.count
            self.ssq += delt * (value - self.mean_est)
            if self.window == self.buffer.size():
                self.count -= 1
                remove = self.buffer.front()
                self.buffer.pop()
                delt = remove - self.mean_est
                self.mean_est -= delt / self.count
                self.ssq -= (delt * (remove-self.mean_est))
            self.buffer.push(value)


cdef class SortinoFixedWindowC(SortinoFixedWindowB):
    """
    See Sortinofixedwindow A and B
    The difference is that the rewards are not squared based on a neg sign.
    The std is still based only on negative rewards.
    Closest to the original Sortino Ratio.
    """

    cpdef double stream(self, double reward):
        """ Adjusts internal metric estimates and returns std estimate """
        self.update(reward)  # done after tail adjust so end doesn't get lost
        if self.buffer.size() <= 1:
            return 0.
        return reward / sqrt((self.ssq+1e-8) / self.count)


class SharpeEWMA(RewardShaper):
    """
    Streams in rewards and returns sharpe ratio values based on an
    exponentially weighted moving average estimate of standard deviation

    If sequential rewards were an array, this would be equivalent to:
        pd.Series(rewards).ewm(alpha=alpha).std()
    """
    def __init__(self, window: int):
        self.alpha = 2 / (float(window + 1))
        self.reset()

    @classmethod
    def from_config(cls, config):
        window = config.reward_shape_window
        return cls(window)

    def reset(self):
        self.count = 0
        self.ewma = 0.
        self.ewma_old = 0.
        self.ewssq_old = 0.
        self.ewssq = 0.
        self.std_est = 0.
        self.w1 = 1.
        self.w2 = 1.

    def stream(self, reward):
        self.update(reward)
        if self.count <= 1:
            return 0.
        return reward / sqrt(self.ewssq)

    # def update(self, value: float):
    #     self.count += 1
    #     delt = value - self.mean_est
    #     self.ewma = self.ewma * (1-self.alpha) + (delt / self.count) * self.alpha
    #     self.ewssq += delt * (value - self.mean_est)

    def update(self, value: float):
        # if self.count == 0:
        #     self.ewma_old = value
        #     self.ewma = value
        self.count += 1
        self.w1 += (1-self.alpha) ** self.count
        self.w2 += ((1-self.alpha) ** self.count) ** 2
        ewma_prev = self.ewma
        self.ewma_old = self.ewma_old * (1-self.alpha) + value
        self.ewma = self.ewma_old / self.w1
        self.ewssq_old = self.ewssq_old * (1-self.alpha) + \
            ((value - self.ewma) * (value - ewma_prev))
        self.ewssq = self.ewssq_old / (self.w1 - self.w2/self.w1)
