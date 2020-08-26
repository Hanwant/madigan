import itertools as it
import numpy as np
from madigan.synth import multi_sine_gen
from arena.portfolio import Portfolio
from arena.broker import Broker
from arena.orders import MarketOrder
from arena.utils import EventType, OrderSide, OrderEntry, OrderTiming, OrderType

class Synth:
    def __init__(self, generator=None, init_cash=1_000_000, levarage=1, discrete_actions=False, action_atoms=11,
                 action_mode="trade", min_lot=100_000, initial_margin=1., slippage_pct=0.001, transaction_cost=0.01,
                 **params):
        assert action_atoms % 2 != 0, "action_atoms must be an odd number - to allow for the action of hold"
        assert action_mode in ("trade", "target")
        if generator is None:
            freq = [1., 2., 3., 4.,]
            mu = [2., 3, 4., 5.] # Keeps negative prices from ocurring
            amp = [1., 2., 3., 4.]
            phase = [0., 1., 2., 0.]
            params = np.stack([freq, mu, amp, phase], axis=1)
            generator = multi_sine_gen(params, dx=0.01)
        self.data_generator, self.test_gen = it.tee(generator)
        obs = next(self.test_gen)
        self._obs_shape = obs.shape
        self.nassets = obs.shape[0]
        self.assets = (f'synth{i}' for i in range(self.nassets))
        self._cash = init_cash
        self.levarage = levarage
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
        self.continuous_actions = not discrete_actions
        self.action_atoms = action_atoms
        self.action_mode = action_mode
        self.min_lot = min_lot
        if not self.discrete_actions:
            raise NotImplementedError("Continuous Actions are not Implemented yet")

    def render(self):
        return self.preprocess(self._current_prices)

    def reset(self):
        self._current_prices = next(self.data_generator)
        return self.get_state()

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

    @property
    def action_space(self):
        """
        Continuous
        1 action for each asset determining direction and amount
        """
        if self.discrete_actions:
            return self.action_space_discrete
        return self.action_space_continuous

    @property
    def action_space_discrete(self):
        """
        Discrete
        3 actions for each asset:
        0: Do Nothing
        1: Buy
        2: Sell
        """
        return self.nassets

    @property
    def action_space_continuous(self):
        return self.nassets


    def step(self, actions):
        """
        if self.discrete_actions:
        action is a matrix (nassets, action_atoms)
        if self.continuous_actions:
        action is vector (nassets) of either change in positions or target portfolio
        """
        assert len(actions) == self.nassets

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
        discretized by self.min_lot
        if units_to_buy >= self.cash :
            ignore_action.
        """
        lot_units = actions - self.action_atoms // 2
        lots = lot_units * self.min_lot
        slippage = self._current_prices * self.slippage_pct * (1 - 2*(lots<0))
        transaction_prices = self._current_prices + slippage
        amounts = lots * transaction_prices
        transaction_prices = []
        transaction_costs = []
        # import ipdb; ipdb.set_trace()
        for i, amount in enumerate(amounts):
            if 0. < abs(amount) < self.available_margin:
                current_price = self._current_prices[i]
                slippage = current_price * self.slippage_pct * (1 - 2*(amount<0))
                transaction_price = current_price + slippage
                # units_to_buy = amount / transaction_price
                self._cash -= amount + abs(slippage) + self.transaction_cost
                self._portfolio[i] += amount / current_price
                transaction_prices.append(transaction_price)
                transaction_costs.append(self.transaction_cost)
            else:
                transaction_prices.append(None)
                transaction_costs.append(None)
        # print(transaction_prices)
        return transaction_prices, transaction_costs

    def step_continuous(self, actions):
        return None, None

    # def buy_order(self, asset, amount, price):
    #     stake = self.portfolio.margin * amount
    #     amount_in_asset_units = abs(stake / price)
    #     return MarketOrder(OrderSide.Buy, asset, amount_in_asset_units, price, OrderEntry.Entry, self.gen_order_id(),
    #                        entry_timing=OrderTiming.NextTick)

    # def sell_order(self, asset, amount, price):
    #     stake = self.portfolio.margin * amount
    #     amount_in_asset_units = abs(stake / price)
    #     return MarketOrder(OrderSide.Sell, asset, amount_in_asset_units, price, OrderEntry.Entry, self.gen_order_id(),
    #                        entry_timing=OrderTiming.NextTick)

    # def action_to_order(self, asset, amount: float, price: float):
    #     stake = self.portfolio.margin * amount
    #     amount_in_asset_units = abs(stake / price)
    #     order = None
    #     if abs(amount) > self.order_eps:
    #         side = OrderSide.Buy if amount > 0. else OrderSide.Sell
    #         order = MarketOrder(side, asset, amount_in_asset_units, price, OrderEntry.Entry, self.gen_order_id(),
    #                         entry_timing=OrderTiming.NextTick)
    #     return order

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    discrete_actions = True
    action_atoms = 11
    env = Synth(discrete_actions=discrete_actions, action_atoms=action_atoms)
    state = env.reset()
    eq = []
    returns = []
    for i in range(100):
        action = np.random.randint(0, action_atoms, env.action_space)
        state, reward, done, info = env.step(action)
        print('eq:', env.equity, 'reward:', reward)
        eq.append(env.equity)
        returns.append(reward)
    fig, ax = plt.subplots()

    ax.plot(range(len(eq)), eq/eq[0], 'blue', label='equity')
    ax.set_ylabel('eq')
    ax2 = ax.twinx()
    ax2.plot(range(len(returns)), returns, 'green', label='returns')
    ax2.set_ylabel('return')
    fig.legend(labels=('eq', 'return'), loc='upper left')
    plt.show()

