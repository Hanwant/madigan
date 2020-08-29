import itertools as it
from collections import deque
from random import random
import numpy as np
from madigan.synth import multi_sine_gen
from madigan.utils import load_json, save_json
from madigan.environments.env import Env



class Synth(Env):
    def __init__(self, generator_params=None, min_tf=1, **params):
        if generator_params is None:
            freq = [1., 2., 3., 4.,]
            mu = [2., 3, 4., 5.] # Keeps negative prices from ocurring
            amp = [1., 2., 3., 4.]
            phase = [0., 1., 2., 0.]
            state_space = np.stack([freq, mu, amp, phase], axis=1)
        else:
            if 'state_space' not in generator_params.keys():
                raise ValueError('generator params needs state_space key')
            state_space = np.array(generator_params['state_space'])

        self.min_tf = min_tf
        self._current_state = deque(maxlen=min_tf)
        super().__init__(self._generator(state_space, dx=0.01), **params)

    def _generator(self, state_space, dx=0.01):
        gen = multi_sine_gen(state_space, dx=dx)
        while True:
            yield {'prices': next(gen), 'timestamps': None}

    def reset(self):
        for i in range(self.min_tf):
            self._current_state = self.preprocess(next(self._data_stream)['prices'])
        return self.current_state

    def get_next_state(self):
        state = self.preprocess(next(self._data_stream)['prices'])
        return state

    @property
    def current_state(self):
        return np.array(self._current_state)

    def preprocess(self, prices):
        self._current_state.append(np.concatenate([prices, self.portfolio_norm]))
        return np.array(self._current_state)



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


def test_env(params, agent, eps=1.):
    env = Synth(**params)
    state = env.reset()
    eq = []
    returns = []
    prices = []
    for i in range(params['nsteps']):
        if random() < eps:
            action = env.action_space.sample()
        else:
            action = agent(state)
        state, reward, done, info = env.step(action)
        if done:
            print('Blown Out')
            break
        # print('eq:', env.equity, 'reward:', reward)
        eq.append(env.equity)
        returns.append(reward)
        prices.append(env.current_prices)
    return {'eq': eq, 'returns': returns, 'prices': np.array(prices)}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    # from madigan.dash.dash_synth import run_dash
    from madigan.dash.dash_synth1 import run_dash

    freq=[1., 2., 3., 4.]
    mu=[2., 3, 4., 5.] # (mu == offset) Keeps negative prices from ocurring
    amp=[1., 2., 3., 4.]
    phase=[0., 1., 2., 0.]
    generator_params = np.stack([freq, mu, amp, phase], axis=1)


    model_config = {
        'model_class': 'conv',
        'd_model': 256,
        'n_layers': 4,
        'in_shape': 4,
        'out_shape': (4, 11),
    }
    agent_config = {
        'type': 'DQN',
        'savepath': '/home/hemu/madigan/farm/',
        'model_params': model_config
    }

    exp_config = dict(
        name='test',
        generator_params={'type': 'multisine',
                          'state_space': generator_params.tolist()},
        discrete_actions=True,
        discrete_action_atoms=11,
        nsteps=100,
        lot_unit_value=10_000,
        agent_params=agent_config,
    )

    param_path = Path('/home/hemu/madigan/madigan/environments')/'test.json'
    save_json(exp_config, param_path)
    data = test_env(exp_config, agent=None, eps=1.)
    # import ipdb; ipdb.set_trace()
    # plot_metrics(data)
    run_dash(data)

