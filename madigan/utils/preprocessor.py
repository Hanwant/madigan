from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import numba

from rollers import Roller as _Roller
from ..utils.data import State
# from ..environments.cpp import State as StateA


def make_preprocessor(config):
    """
    Choices for config.preprocessor_type:
    StackerDiscrete (== WindowedStacker):  maintains a fixed size window of obs
    StackerContinuous: maintains a variable sized window of a fixed time offset
    RollerDiscrete: Rolling summary functions for discrete (fixed) windows
    RollerContinuous: Rolling functions for continuous (variable) time-windows
    """
    if config.preprocessor_type in ("WindowedStacker", "StackerDiscrete",
                                    "StackerDiscreteReturns"):
        return StackerDiscrete.from_config(config)
    elif config.preprocessor_type in ("StackerContinuous"):
        return StackerContinuous.from_config(config)
    elif config.preprocessor_type in ("RollerDiscrete", ):
        return RollerDiscrete.from_config(config)
    elif config.preprocessor_type in ("CustomA", ):
        return CustomA.from_config(config)
    raise NotImplementedError(
        f"{config['preprocessor_type']} is not implemented ")


def make_normalizer(norm_type):
    """
    Choices

    lookback: norm everything by current price x / x[-1]
    lookback_log: log(x / x[-1])
    standard_normal: (x-x.mean()) / x.std()
    expanding: norm by an expanding window mean
    """
    if norm_type == 'lookback':
        return lambda x: x / x[-1]
    if norm_type == 'lookback_log':
        return lambda x: np.log(x / x[-1])
    if norm_type == 'standard_normal':
        return lambda x: (x-x.mean()) / x.std()
    if norm_type == 'expanding':
        return lambda x: x / _expanding_mean(x)

def standard_norm(x):
    """
    Nan safe version of standard normalization
    otherwise a lambda is enough I.e lambda x: (x-x.mean()) / x.std()
    """
    mean = x.mean()
    return (x-mean) / np.nan_to_num(x.std(), mean)

class PreProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def stream(self, data):
        pass

    @abstractmethod
    def current_data(self):
        pass

    @classmethod
    def from_config(cls, config):
        return make_preprocessor(config)

    @abstractmethod
    def initialize_history(self):
        pass

class StackerDiscrete(PreProcessor):
    def __init__(self, window_len, norm=True, norm_type='lookback'):
        self.k = window_len
        self.min_tf = self.k
        self.norm = norm
        self.norm_fn = make_normalizer(norm_type)
        self.price_buffer = deque(maxlen=self.k)
        self.portfolio_buffer = deque(maxlen=self.k)
        self.time_buffer = deque(maxlen=self.k)
        self.feature_output_shape = (self.k, 1)

    @classmethod
    def from_config(cls, config):
        pconf = config.preprocessor_config
        norm = pconf.norm if 'norm' in pconf.keys() else False
        norm_type = pconf.norm_type if 'norm_type' in pconf.keys() else None
        return cls(pconf.window_length, norm, norm_type)

    def __len__(self):
        return len(self.price_buffer)

    def stream_state(self, state):
        self.price_buffer.append(np.array(state.price, copy=True))
        self.portfolio_buffer.append(np.array(state.portfolio, copy=True))
        self.time_buffer.append(np.array(state.timestamp, copy=True))

    def stream(self, data):
        if isinstance(data, tuple):
            self.stream_state(data[0])  # assume srdi
        else:  # assume data is State
            self.stream_state(data)

    def current_data(self):
        price = np.array(self.price_buffer, copy=True)
        if self.norm:
            price = self.norm_fn(price)
        portfolio = np.array(self.portfolio_buffer, copy=True)
        timestamp = np.array(self.time_buffer, copy=True)
        return State(price, portfolio, timestamp)


    def initialize_history(self, env):
        while len(self) < self.k:
            _state, reward, done, info = env.step()
            self.stream_state(_state)

    def reset_state(self):
        self.price_buffer.clear()
        self.portfolio_buffer.clear()
        self.time_buffer.clear()

class StackerDiscreteReturns(StackerDiscrete):
    def current_data(self):
        price = np.array(self.price_buffer, copy=True)
        if self.norm:
            price = self.norm_fn(price)
        price = np.diff(price)
        portfolio = np.array(self.portfolio_buffer, copy=True)[1:]
        timestamp = np.array(self.time_buffer, copy=True)[1:]
        return State(price, portfolio, timestamp)


class StackerContinuous(StackerDiscrete):
    def __init__(self, window_len, norm=True, norm_type='lookback'):
        self.timeframe = window_len
        self.min_tf = self.k
        self.norm = norm
        self.norm_fn = make_normalizer(norm_type)
        self.price_buffer = deque()
        self.portfolio_buffer = deque()
        self.time_buffer = deque()
        self.current_size = 0

    def stream_state(self, state):
        self.price_buffer.append(np.array(state.price, copy=True))
        self.portfolio_buffer.append(np.array(state.portfolio, copy=True))
        self.time_buffer.append(np.array(state.timestamp, copy=True))
        self.current_size += 1
        if self.current_size > 1:
            # Test difference in real timestamps
            # Generalized to indexes instead of actual timestamps
            while (self.time_buffer[-1] -
                   self.time_buffer[0]) > self.timeframe:
                self.portfolio_buffer.popleft()
                self.price_buffer.popleft()
                self.time_buffer.popleft()


class RollerDiscrete(PreProcessor):
    """
    Wraps Roller to accumulate rolling window features
    using discrete windows for the final aggregation of features
    Single Price Series.
    """
    def __init__(self, timeframes: list, window_len=64):
        self.timeframes = timeframes
        self.k = window_len
        self._roller = _Roller(timeframes)
        self.min_tf = max(self._roller.timeframes_uint64_t)
        self.price_buffer = deque(maxlen=self.k)
        self.feature_buffer = deque(maxlen=self.k)
        self.portfolio_buffer = deque(maxlen=self.k)
        self.time_buffer = deque(maxlen=self.k)
        n_feats = 8*len(self.timeframes) + 1  # 8 from roller, 1 for close price
        self.feature_output_shape = (window_len, n_feats)

    @classmethod
    def from_config(cls, config):
        window_len = config.preprocessor_config.window_length
        timeframes = config.preprocessor_config.timeframes
        return cls(timeframes, window_len)

    def __len__(self):
        return len(self.price_buffer)

    def stream_state(self, state):
        self.price_buffer.append(state.price)
        self.portfolio_buffer.append(state.portfolio)
        self.time_buffer.append(state.timestamp)
        feats = np.nan_to_num(
            self._roller.roll(state.price, np.array([state.timestamp])))
        self.feature_buffer.append(feats)

    def stream_srdi(self, srdi):
        self.stream_state(srdi[0])

    def stream(self, data):
        if isinstance(data, tuple):
            self.stream_srdi(data)  # assume srdi
        else:  # assume data is State
            self.stream_state(data)

    def current_data(self):
        current_len = len(self.price_buffer)
        prices = np.array(self.price_buffer, copy=True)
        current_price = prices[-1, 0]
        feats = np.concatenate(self.feature_buffer, axis=0)
        feats[:, :4, :] /= current_price
        prices /= current_price
        feats = feats.reshape(current_len, -1)
        feats = np.concatenate([prices, feats], axis=1)
        return State(feats,
                     np.array(self.portfolio_buffer),
                     np.array(self.time_buffer))

    def initialize_history(self, env):
        while len(self) < self.k:
            _state, reward, done, info = env.step()
            self.stream_state(_state)

    def reset_state(self):
        self.price_buffer.clear()
        self.portfolio_buffer.clear()
        self.time_buffer.clear()
        self.feature_buffer.clear()


class CustomA(StackerDiscrete):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.feature_output_shape = (self.k-1, 2)

    def current_data(self):
        price = np.array(self.price_buffer, copy=True).squeeze()
        if self.norm:
            price = self.norm_fn(price)
        returns = (price[1:] - price[:-1])
        pct_up = np.empty_like(returns)
        _expanding_mean(returns > 0., pct_up)
        price = np.stack([price[1:], pct_up], axis=-1)
        # import ipdb; ipdb.set_trace()
        portfolio = np.array(self.portfolio_buffer, copy=True)
        timestamp = np.array(self.time_buffer, copy=True)
        return State(price, portfolio, timestamp)


class AutoEncoder(PreProcessor):
    def __init__(self, ):
        pass

    @abstractmethod
    def stream(self, data):
        pass

    @abstractmethod
    def current_data(self):
        pass

    @classmethod
    def from_config(cls, config):
        return make_preprocessor(config)

    @abstractmethod
    def initialize_history(self):
        pass


@numba.guvectorize([(numba.float64[:], numba.float64[:])],
                   '(n)->(n)', nopython=True)
def _expanding_mean(arr, ma):
    """ expanding/running mean - equiv to pd.expanding().mean()"""
    n = arr.shape[0]
    # ma = np.empty(n)
    w = 1
    if not np.isnan(arr[0]):
        ma_old = arr[0]
    else:
        ma_old = 0.
    ma[0] = ma_old
    for i in range(1, n):
        if not np.isnan(arr[i]):
            ma_old = ma_old + arr[i]
            w += 1
        ma[i] = ma_old / w
