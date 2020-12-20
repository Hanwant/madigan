import itertools as it
from collections import deque
from random import random
import numpy as np
from ..synth import multi_sine_gen
from ..utils.config import load_config, save_config
from ..utils.data import State, SARSD
from .env import Env, EnvPy

def default_params():
    param = {'type': 'multisine'}
    freq = [1., 0.3, 2., 0.5,]
    mu = [2., 2.1, 2.2, 2.3] # Keeps negative prices from ocurring
    amp = [1., 1.2, 1.3, 1.]
    phase = [0., 1., 2., 1.]
    param['state_space'] = np.stack([freq, mu, amp, phase], axis=1)
    return param


class Synth(EnvPy):
    """
    DEPRECATED
    Uses numba for core computation (multi_sine_gen)
    """
    def __init__(self, generator_params=None, min_tf=1, **config):
        if generator_params is None:
            generator_params = default_params()
        else:
            if 'state_space' not in generator_params.keys():
                raise ValueError('Synth generator params needs state_space key')
        state_space = np.array(generator_params['state_space'])
        self.min_tf = min_tf
        self._current_state = deque(maxlen=min_tf)
        super().__init__(self._generator(state_space, dx=0.01), **config)

    @staticmethod
    def _generator(state_space, dx=0.01):
        gen = multi_sine_gen(state_space, dx=dx)
        while True:
            yield {'prices': next(gen), 'timestamps': None}

    def reset(self):
        self.reset_portfolio()
        for i in range(self.min_tf):
            current_state = self.preprocess(next(self._data_stream)['prices'])
        return current_state

    def get_next_state(self):
        state = self.preprocess(next(self._data_stream)['prices'])
        return state

    @property
    def current_state(self):
        return State(price=np.array(self._current_state), port=self.portfolio_norm)

    def preprocess(self, prices):
        normed_prices = prices
        self._current_state.appendleft(normed_prices)
        return State(price=np.array(self._current_state), port=self.portfolio_norm)


def plot_metrics(data):
    eq, returns = data['eq'], data['returns']
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(range(len(eq)), eq/eq[0], 'blue', label='equity')
    ax[0].set_ylabel('eq')
    ax_0 = ax[0].twinx()
    ax_0.plot(range(len(returns)), returns, 'green', label='returns')
    ax_0.set_ylabel('return')
    ax[1].plot(np.array(data['prices']))
    fig.legend(labels=('eq', 'return'), loc='upper left')
    plt.show()


def test_env(exp_config, agent, eps=1.):
    env = Synth(**exp_config)
    state = env.reset()
    eq = []
    returns = []
    prices = []
    positions = []
    cash = []
    margin = []
    for i in range(exp_config['nsteps']):
        if random() < eps:
            action = env.action_space.sample()
        else:
            action = agent(state)
        state, reward, done, info = env.step(action)
        if done:
            print(info['Event'])
            break
        # print('eq:', env.equity, 'reward:', reward)
        eq.append(env.equity)
        returns.append(reward)
        prices.append(env.current_prices)
        positions.append(env.portfolio)
        cash.append(env.cash)
        margin.append(env.available_margin)
    return {'eq': eq, 'returns': returns, 'prices': np.array(prices), 'positions': positions,
            'assets': env.assets, 'cash': cash, 'margin': margin}

