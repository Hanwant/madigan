import itertools as it
import numpy as np

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

class Env:
    """
    Base Class for trading environments.

    Implements accounting and conversion of model outputs to trading actions
    Relies on child classes implementing get_state(self) and prepocess(self)

    Treats the output of next(data_feed) (obtained at init) as an array of prices
    for n assets, to which self.preprocess is applied along with self.get_state()
    """
    def __init__(self,
                 data_stream,
                 init_cash=1_000_000,
                 levarage=1,
                 discrete_actions=False,
                 discrete_action_atoms=11,
                 action_mode="trade",
                 lot_unit_value=100_000,
                 initial_margin=1.,
                 slippage_pct=0.001,
                 transaction_cost=0.01,
                 assets = None,
                 maintenance_margin=0.1,
                 **params):
        assert discrete_action_atoms % 2 != 0, "action_atoms must be an odd number - to allow for the action of hold"
        assert action_mode in ("trade", "target")

        self._data_stream, self.test_stream = it.tee(data_stream)
        self._cash = init_cash
        self.levarage = levarage
        self._borrowed_for_short = 0.
        self.initial_margin = initial_margin
        self.slippage_pct = slippage_pct
        self.transaction_cost = transaction_cost
        self._maintenance_margin = maintenance_margin
        self.state = []

        # self.portfolio.= Portfolio(deposit=init_cash, denom="NZD", assets=self.assets,
        #                       levarage=1., abs_comm=0., rel_comm=0.01)
        # self.broker = Broker(porfolios=(self.portfolio. ))

        # Getting number of assets and state observation shape after preprocessing
        prices = next(self.test_stream)['prices']
        self._current_prices = prices
        self.nassets = prices.shape[0]
        self.assets = assets or [f'asset_{i}' for i in range(self.nassets)]
        self._portfolio = np.zeros(self.nassets, np.float64)

        self.current_order_id = 0
        self.order_eps = 0.01
        self.discrete_actions = discrete_actions
        if self.discrete_actions:
            self.continuous_actions = not discrete_actions
            self.discrete_action_atoms = discrete_action_atoms
            self._action_space = DiscreteActionSpace((0, discrete_action_atoms), self.nassets)
            self.action_mode = action_mode
            self.lot_unit_value = lot_unit_value
        else:
            raise NotImplementedError("Continuous Actions are not Implemented yet")
        # Doing this at the end as preprocessing should require self.portfolio (as input to model)
        # self._obs_shape = self.preprocess(prices).shape

    def render(self):
        raise NotImplementedError

    def reset(self):
        # self._current_prices = next(self._data_stream)
        # return self.get_state()
        raise NotImplementedError

    @property
    def observation_shape(self):
        return None

    @property
    def action_space(self):
        return self._action_space

    @property
    def current_prices(self):
        return self._current_prices

    @property
    def current_state(self):
        """
        An example implementation:
            return self.preprocess(self._current_prices)
        """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def preprocess(self, data):
        raise NotImplementedError

    @property
    def cash(self):
        return self._cash

    @property
    def portfolio(self):
        """ Human Readable """
        return self._portfolio.copy()

    @property
    def portfolio_h(self):
        """ Human Readable """
        return dict(zip(self.assets, self._portfolio))

    @property
    def equity(self):
        """ Total value including cash + live value of assets"""
        return self._cash + (self._current_prices * self._portfolio).sum()

    @property
    def maintenance_margin(self):
        return self._maintenance_margin

    @property
    def available_margin(self):
        return self._cash + np.sum((self._portfolio * self._current_prices)[self._portfolio < 0])

    @property
    def portfolio_norm(self):
        """ Portfolio normalized by total equity value"""
        # eq_values = np.concatenate([[self._cash], self._current_prices * self._portfolio])
        eq_values = self._current_prices * self._portfolio
        return eq_values / (eq_values.sum() + 1e-8) # addition to prevent div by 0 at beginning

    @property
    def holdings(self):
        """ Portfolio in vector form """
        return self._portfolio

    def gen_order_id(self):
        self.current_order_id += 1
        return self.current_order_id

    def check_risk(self, asset_id: int, amount):
        """
        Returns True if the transaction can be made
        """
        if amount == 0.:
            return False
        if amount > 0.:
            if self._portfolio[asset_id] < 0.:
                return True
            if amount < self.available_margin:
                return True
        if self._portfolio[asset_id] > 0.:
            return True
        if abs(amount) < self.available_margin:
            return True

    def step(self, actions):
        """
        if self.discrete_actions:
        action is a matrix (nassets, action_atoms)
        if self.continuous_actions:
        action is vector (nassets) of either change in positions or target portfolio
        """
        assert len(actions) == self.nassets
        if isinstance(actions, list):
            actions = np.array(actions)

        if self.equity <= 0.:
            data = next(self._data_stream)
            self._current_prices = data['prices']
            return self.preprocess(self._current_prices), 0., True, {'Event': "BLOWOUT", 'transaction_price': None, 'transaction_cost': None}

        if self.available_margin < self.maintenance_margin:
            data = next(self._data_stream)
            self._current_prices = data['prices']
            return self.preprocess(self._current_prices), 0., True, {"Event": "MARGINCALL", 'transaction_price': None, 'transaction_cost': None}

        info = {}
        done = False
        equity_current_state = self.equity

        if self.discrete_actions:
            transaction_price, transaction_cost = self.step_discrete(actions)
        else:
            transaction_price, transaction_cost = self.step_continuous(actions)
        info['tranaction_price'] = transaction_price
        info['transaction_cost'] = transaction_cost

        data = next(self._data_stream)
        self._current_prices = data['prices']

        equity_next_state = self.equity
        immediate_return = (equity_next_state / equity_current_state) - 1

        return self.preprocess(self._current_prices), immediate_return, done, info

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
            if self.check_risk(i, amount):
                self.transaction(i, amount, transaction_prices[i], transaction_costs[i])
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
