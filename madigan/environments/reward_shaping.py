"""
Contains classes for transforming raw rewards (assumed to be log returns).
transformations include:
- sharpe ratio
- sortino ratio
"""
from abc import ABC, abstractmethod
from collections import deque
import math


def make_reward_shaper(config):
    name = config['reward_shaper']
    if name in globals():
        return globals()[name].from_config(config)
    raise NotImplementedError(f"reward_shaper {name} has not "
                              "been implemented")

class RewardShaper(ABC):
    """
    May keep a running estimate of metrics such as mean/var/std.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        """
        Most likely entry point for initialization
        """

    @abstractmethod
    def reset(self):
        """
        reset running estimates
        """

    @abstractmethod
    def stream(self, reward):
        """
        Streams in new reward and returns a transformation
        """


class SharpeFixedWindow(RewardShaper):
    """
    Streams in rewards and returns sharpe ratio values based on a rolling
    window estimate of standard deviation

    If sequential rewards were an array, this would be equivalent to:
        pd.Series(rewards).rolling(window, min_periods=2).std(ddof=0)
    """
    def __init__(self, window: int):
        self.window = window
        self.buffer = deque(maxlen=window)
        self.reset()

    @classmethod
    def from_config(cls, config):
        window = config.reward_shape_window
        return cls(window)

    def reset(self):
        """ Needs to be called when environment resets / episode ends. """
        self.buffer.clear()
        self.mean_est = 0.
        self.count = 0
        self.ssq = 0.

    def stream(self, reward):
        """ Adjusts internal metric estimates and returns std estimate """
        if len(self.buffer) == self.window:
            self.tail_adjust()
        self.head_add(reward)  # done after tail adjust so end doesn't get lost
        if len(self.buffer) <= 1:
            return 0.
        return reward / math.sqrt(self.ssq / self.count)

    def head_add(self, value: float):
        self.count += 1
        delt = value - self.mean_est
        self.mean_est += delt / self.count
        self.ssq += delt * (value - self.mean_est)
        self.buffer.append(value)

    def tail_adjust(self):
        self.count -= 1
        remove = self.buffer.popleft()
        delt = remove - self.mean_est
        self.mean_est -= delt / self.count
        self.ssq -= (delt * (remove-self.mean_est))


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
        return reward / math.sqrt(self.ewssq)

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

