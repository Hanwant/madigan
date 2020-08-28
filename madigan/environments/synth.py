import itertools as it
from random import random
import numpy as np
from madigan.synth import multi_sine_gen
from madigan.utils import load_json, save_json
from arena.portfolio import Portfolio
from arena.broker import Broker
from arena.orders import MarketOrder
from arena.utils import EventType, OrderSide, OrderEntry, OrderTiming, OrderType

class DiscreteActionSpace:
    def __init__(self, ranges: tuple, n: int):
        assert len(ranges) == 2
        self.ranges = ranges
        self.low = ranges[0]
        self.high = ranges[1]
        self.n = n

    def sample(self):
        action = np.random.randint(self.low, self.high, self.n)
        return action

    @property
    def shape(self):
        return (self.n, )


class Synth:
    def __init__(self, generator_params=None, init_cash=1_000_000, levarage=1, discrete_actions=False, discrete_action_atoms=11,
                 action_mode="trade", lot_unit_value=100_000, initial_margin=1., slippage_pct=0.001, transaction_cost=0.01,
                 **params):
        assert discrete_action_atoms % 2 != 0, "action_atoms must be an odd number - to allow for the action of hold"
        assert action_mode in ("trade", "target")
        if generator_params is None:
            freq = [1., 2., 3., 4.,]
            mu = [2., 3, 4., 5.] # Keeps negative prices from ocurring
            amp = [1., 2., 3., 4.]
            phase = [0., 1., 2., 0.]
            params = np.stack([freq, mu, amp, phase], axis=1)
        else:
            if 'state_space' not in generator_params.keys():
                raise ValueError('generator params needs state_space key')
            params = np.array(generator_params['state_space'])

        generator = multi_sine_gen(params, dx=0.01)
        self.data_generator, self.test_gen = it.tee(generator)
        obs = next(self.test_gen)
        self._obs_shape = obs.shape
        self.nassets = obs.shape[0]
        self.assets = (f'synth{i}' for i in range(self.nassets))
        self._cash = init_cash
        self.levarage = levarage
        self._borrowed_for_short = 0.
        self.initial_margin = initial_margin
        self.slippage_pct = slippage_pct
        self.transaction_cost = transaction_cost
        self.state = []
        self._portfolio = np.zeros(self.nassets, np.float64)
        self._current_prices = obs
        # self.portfolio.= Portfolio(deposit=init_cash, denom="NZD", assets=self.assets,
        #                       levarage=1., abs_comm=0., rel_comm=0.01)
        # self.broker = Broker(porfolios=(self.portfolio. ))

        self.current_order_id = 0
        self.order_eps = 0.01
        self.discrete_actions = discrete_actions
        if self.discrete_actions:
            self.continuous_actions = not discrete_actions
            self.discrete_action_atoms = discrete_action_atoms
            self.action_space = DiscreteActionSpace((0, discrete_action_atoms), self.nassets)
            self.action_mode = action_mode
            self.lot_unit_value = lot_unit_value
        else:
            raise NotImplementedError("Continuous Actions are not Implemented yet")

    def render(self):
        return self.preprocess(self._current_prices)

    def reset(self):
        self._current_prices = next(self.data_generator)
        return self.get_state()

    @property
    def current_prices(self):
        return self._current_prices

    def get_state(self):
        state = np.concatenate([self.preprocess(self._current_prices), self.portfolio_norm])
        return state

    def preprocess(self, data):
        return data.copy()

    @property
    def cash(self):
        return self._cash

    @property
    def portfolio(self):
        """ Human Readable """
        return dict(zip(self.assets, self._portfolio))

    @property
    def equity(self):
        """ Total value including cash + live value of assets"""
        return self._cash + (self._current_prices * self._portfolio).sum()

    @property
    def available_margin(self):
        return self._cash + np.sum((self._portfolio * self._current_prices)[self._portfolio < 0])

    @property
    def portfolio_norm(self):
        """ Portfolio normalized by total equity value"""
        eq_values = np.concatenate([[self._cash], self._current_prices * self._portfolio])
        return eq_values / eq_values.sum()

    @property
    def holdings(self):
        """ Portfolio in vector form """
        return self._portfolio

    def gen_order_id(self):
        self.current_order_id += 1
        return self.current_order_id

    @property
    def observation_shape(self):
        return self._obs_shape

    def step(self, actions):
        """
        if self.discrete_actions:
        action is a matrix (nassets, action_atoms)
        if self.continuous_actions:
        action is vector (nassets) of either change in positions or target portfolio
        """
        assert len(actions) == self.nassets

        if self.equity <= 0.:
            self._current_prices = next(self.data_generator)
            equity_next_state = self.equity
            return self.get_state(), 0., True, {'transaction_price': None, 'transaction_cost': None}

        info = {}
        done = False
        equity_current_state = self.equity

        if self.discrete_actions:
            transaction_price, transaction_cost = self.step_discrete(actions)
        else:
            transaction_price, transaction_cost = self.step_continuous(actions)
        info['tranaction_price'] = transaction_price
        info['transaction_cost'] = transaction_cost

        self._current_prices = next(self.data_generator)

        equity_next_state = self.equity
        immediate_return = (equity_next_state / equity_current_state) - 1

        return self.get_state(), immediate_return, done, info

    def step_discrete(self, actions):
        """
        actions span from -1. to 1. of cash remaining
        discretized by self.lot_unit_value
        if units_to_buy >= self.cash :
            ignore_action.
        """
        lot_units = actions - self.discrete_action_atoms // 2
        lots = lot_units * self.lot_unit_value
        slippage = self._current_prices * self.slippage_pct * (1 - 2*(lots<0))
        transaction_prices = self._current_prices + slippage
        transaction_costs = [self.transaction_cost] * len(actions) #+ np.abs(slippage)
        amounts = lots * transaction_prices
        # transaction_prices = []
        # transaction_costs = []
        for i, amount in enumerate(amounts):
            if 0. < abs(amount) < self.available_margin: # if not enough margin, earlier orders have precedence
                self.transaction(i, amount, transaction_prices[i], transaction_costs[i])
                # import ipdb; ipdb.set_trace()
            else:
                transaction_prices[i] = None
                transaction_costs[i] = None
        # print(transaction_prices)
        return transaction_prices, transaction_costs

    def step_continuous(self, actions):
        return None, None

    def transaction(self, asset_idx, cash_amount, transaction_price=None, transaction_cost=None):
        if transaction_price is None or transaction_cost is None:
            current_price = self._current_prices[asset_idx]
            slippage = current_price * self.slippage_pct * (1 - 2*(cash_amount<0))
        transaction_price = transaction_price or (current_price + slippage)
        transaction_cost = transaction_cost or self.transaction_cost
        units_to_buy = cash_amount / transaction_price
        self._cash -= cash_amount + transaction_cost
        self._portfolio[asset_idx] += units_to_buy



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
    from madigan.dash.dash_synth import run_dash
    from madigan.dash.dash_synth1 import run_dash

    freq=[1., 2., 3., 4.]
    mu=[2., 3, 4., 5.] # Keeps negative prices from ocurring
    amp=[1., 2., 3., 4.]
    phase=[0., 1., 2., 0.]
    generator_params = np.stack([freq, mu, amp, phase], axis=1)

    params = dict(
        name='test',
        generator_params={'type': 'multisine',
                          'state_space': generator_params.tolist()},
        discrete_actions=True,
        discrete_action_atoms=11,
        nsteps=100,
        lot_unit_value=10_000
    )
    import json
    import os
    from pathlib import Path
    param_path = Path('/home/hemu/madigan/madigan/environments')/'test.json'
    with open(param_path, 'w') as f:
        json.dump(params, f)
    data = test_env(params, agent=None, eps=1.)
    # import ipdb; ipdb.set_trace()
    # plot_metrics(data)
    run_dash(data)

