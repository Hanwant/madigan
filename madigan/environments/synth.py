import itertools as it
from collections import deque
from random import random
import numpy as np
from ..synth import multi_sine_gen
from ..utils.config import load_config, save_config
from ..utils.data import State, SARSD
from .env import Env

def default_params():
    param = {'type': 'multisine'}
    freq = [1., 2., 3., 4.,]
    mu = [2., 3, 4., 5.] # Keeps negative prices from ocurring
    amp = [1., 2., 3., 4.]
    phase = [0., 1., 2., 0.]
    param['state_space'] = np.stack([freq, mu, amp, phase], axis=1)
    return param


class Synth(Env):
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
        self._portfolio = np.zeros(self.nassets)
        self._cash = self.init_cash
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
        self._current_state.append(normed_prices)
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    # from madigan.dash.dash_synth import run_dash
    from madigan.dash.dash_synth1 import run_dash

    freq=[1., 2., 3., 4.]
    mu=[2., 3, 4., 5.] # (mu == offset) Keeps negative prices from ocurring
    amp=[1., 2., 3., 4.]
    phase=[0., 1., 2., 0.]
    gen_state_space = np.stack([freq, mu, amp, phase], axis=1) # (n_assets, nparameters)

    generator_params = {'type': 'multisine',
                          'state_space': gen_state_space.tolist()}
    # Params
    n_assets = generator_params['state_space'].shape[0]
    nfeats = (2 * n_assets)
    min_tf = 1
    in_shape = (min_tf, nfeats)
    discrete_actions = True
    discrete_action_atoms = 11
    nsteps = 100

    model_config = {
        'model_class': 'conv',
        'd_model': 256,
        'n_layers': 4,
        'in_shape': in_shape,
        'out_shape': (4, 11),
    }
    agent_config = {
        'type': 'DQN',
        'savepath': '/home/hemu/madigan/farm/',
        'model_params': model_config
    }
    exp_config = dict(
        name='test',
        generator_params=generator_params,
        discrete_actions=discrete_actions,
        discrete_action_atoms=discrete_action_atoms,
        nsteps=nsteps,
        lot_unit_value=1_000,
        min_tf = min_tf,
        agent_config=agent_config,
    )

    param_path = Path('/home/hemu/madigan/madigan/environments')/'test.json'
    save_config(exp_config, param_path)
    data = test_env(exp_config, agent=None, eps=1.)
    # plot_metrics(data)
    run_dash(data)

