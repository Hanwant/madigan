import pytest
import numpy as np
from numpy import isclose
from numpy.testing import assert_equal, assert_allclose
from madigan.utils import time_profile
from madigan.utils.config import Config
from madigan.environments.synth import Synth as Synth_py
from madigan.environments.cpp import Asset, Assets, RiskInfo
from madigan.environments.cpp import Synth, Portfolio, Account, Broker, Env

config = Config({
    'data_source_type': 'Synth',
    'data_source_config': {
        'freq': [1., 0.3, 2., 0.5],
        'mu': [2., 2.1, 2.2, 2.3],
        'amp': [1., 1.2, 1.3, 1.0],
        'phase': [0., 1.0, 2., 1.],
        'dX': 0.01,
        'noise': 0.
    }
})
assets = Assets(['BTCUSD', 'ETHUSD', 'BTCETH', 'EURUSD'])


def test_assets_init():
    asset1 = Asset("EURUSD")
    asset2 = Asset("GBPUSD")
    assets = Assets(['EURUSD', 'GBPUSD'])
    assets_ = Assets([asset1, asset2])
    assert all((ass1.code == ass2.code for ass1, ass2 in zip(assets, assets_)))


def test_dataSource_init():
    synth1 = Synth()
    synth2 = Synth([1., 0.3, 2., 0.5], [2., 2.1, 2.2, 2.3], [1., 1.2, 1.3, 1.],
                   [0., 1., 2., 1.],
                   dx=0.01,
                   noise=0.)
    synth3 = Synth(dict(config))

    prices1 = synth1.getData()
    prices2 = synth2.getData()
    prices3 = synth3.getData()
    assert isinstance(prices1, np.ndarray)
    assert isinstance(prices2, np.ndarray)
    assert isinstance(prices3, np.ndarray)
    assert_equal(prices1, prices2)
    assert_equal(prices2, prices3)


def test_buffer_referencing():
    synth = Synth()
    dat = synth.getData()  # RETURNS CONST REFERENCE FROM C++ SIDE
    arr_default = np.array(dat)
    arr_ref = np.array(dat, copy=False)
    arr_copy = np.array(dat, copy=True)
    dat[0] = 33333
    assert synth.currentData()[0] == 33333
    assert arr_ref[0] == dat[0], "pybind is not returning by reference"
    assert arr_copy[0] != dat[0]
    if arr_default[0] == dat[0]:
        print("default ndarray constructor references buffer without copy")
    else:

        print("default ndarray constructor copies buffer ")


def test_dataSource_speed():
    synth = Synth()
    dat = synth.getData()
    synth_py = iter(Synth_py(discrete_actions=True)._data_stream)
    time_profile(1000,
                 1,
                 py_synth=lambda: next(synth_py),
                 cppsynth=lambda: synth.getData())


def test_port_init():
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    # port1 = Portfolio()
    port1 = Portfolio("port_1", assets=assets, initCash=1_000_000)
    print("nAssets: ", port1.nAssets)
    print("assets: ", port1.assets)
    print("equity: ", port1.equity)
    print("cash: ", port1.cash)
    assert isinstance(port1.currentPrices, np.ndarray)
    assert isinstance(port1.ledger, np.ndarray)
    # port.currentPrices()


def test_port_data_ref():
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    synth = Synth()
    ref = synth.getData()
    port1 = Portfolio("port_1", assets=assets, initCash=1_000_000)
    port1.setDataSource(synth)
    assert isinstance(port1.currentPrices, np.ndarray)
    assert_equal(synth.getData(), port1.currentPrices)


@pytest.mark.skip("util")
def transaction(units, init_cash, prices, assetIdx=0, margin=1., num_trans=4):
    assert 0. <= margin <= 1., "margin (required initial) must be between 0 and 1"
    cash = init_cash
    price = prices[assetIdx]
    cost = margin * (price * units)
    cash -= cost
    borrowed_margin = (1 - margin) * (price * units)
    used_margin = cost
    new_price = price
    if borrowed_margin < 0.:
        cash -= borrowed_margin
        borrowed_margin = 0.
    equity = cash + units * (price) - borrowed_margin
    return cash, borrowed_margin, equity


@pytest.mark.skip("util")
def compare_port_transaction_ref(units, init_cash, required_margin):
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    data_source = Synth(config)
    assetIdx = 0
    port = Portfolio("port_accounting_test", assets, init_cash)
    port.setDataSource(data_source)
    port.setRequiredMargin(required_margin)

    current_prices = data_source.getData()
    # port calc
    port.handleTransaction(assetIdx, current_prices[assetIdx], units, 0.)
    # reference calc
    cash, borrowed_margin, equity = transaction(units=units,
                                                init_cash=init_cash,
                                                prices=current_prices,
                                                assetIdx=assetIdx,
                                                margin=required_margin)
    assert cash == port.cash
    assert borrowed_margin == port.borrowedMargin
    assert equity == port.equity


def test_port_accounting_logic():
    compare_port_transaction_ref(1000., 1_000_000, 1.)  # buy on cash
    compare_port_transaction_ref(-1000., 1_000_000, 1.)  # sell on cash
    compare_port_transaction_ref(1000., 1_000_000, .1)  # buy on margin (0.1)
    compare_port_transaction_ref(-1000., 1_000_000, .1)  # sell on margin (0.1)


def test_port_ledger():
    synth = Synth()
    synth.getData()
    port1 = Portfolio("port_1", assets=assets, initCash=1_000_000)
    port1.setDataSource(synth)
    port1.setRequiredMargin(1.)
    current_prices = synth.currentPrices()
    for assetIdx, units in zip([0, 1, 2, 3], [1000, 2000, -4000, 1000]):
        port1.handleTransaction(assetIdx, current_prices[assetIdx], units, 0.)
    ATOL = 1e-8
    assert abs((1-port1.ledgerNormed.sum())*port1.equity \
               - (port1.cash - port1.borrowedMargin)) < ATOL
    # Since port1.borrowedMargin should be 0. given full requiredMargin of 1.
    assert abs((1 - port1.ledgerNormed.sum()) * port1.equity -
               port1.cash) < ATOL
    assert abs(port1.ledgerNormedFull.sum() - 1.) < ATOL

    port2 = Portfolio("port_2", assets=assets, initCash=1_000_000)
    port2.setDataSource(synth)
    port2.setRequiredMargin(.1)
    current_prices = synth.currentPrices()
    for assetIdx, units in zip([0, 1, 2, 3], [1000, 2000, -4000, 1000]):
        port2.handleTransaction(assetIdx, current_prices[assetIdx], units, 0.)
    assert abs((1-port2.ledgerNormed.sum())*port2.equity \
               - (port2.cash-port2.borrowedMargin)) < ATOL
    assert abs(port1.ledgerNormedFull.sum() - 1.) < ATOL
    # abs norm ledger
    # ledgerAbsNormed == ledgerNormed / ledgerNormed.abs().sum()
    renormed_port = (port1.ledgerAbsNormedFull * 1 /
                     port1.ledgerAbsNormedFull.sum())
    np.testing.assert_allclose(renormed_port, port1.ledgerNormedFull)


def test_account_init():
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    port = Portfolio("REF PORT", assets, 1_000_000)
    account1 = Account("coinbase_acc", assets=assets, initCash=1_000_000)
    account2 = Account(assets=assets, initCash=1_000_000)
    account3 = Account(port)
    del port  # ACCOUNT DOES NOT REFER TO PORT PASSED IN CONSTRUCTOR
    print("account1 ports", account1.portfolios())
    print("account2 ports", account2.portfolios())
    print("account3 ports", account3.portfolios())


def test_acc_data_ref():
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    synth = Synth()
    ref = synth.getData()
    acc = Account("acc_test_data_ref", assets=assets, initCash=1_000_000)
    acc.setDataSource(synth)
    assert isinstance(acc.currentPrices(), np.ndarray)
    assert_equal(synth.getData(), acc.currentPrices())


def test_acc_port_routing():
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    synth = Synth()
    current_prices = synth.getData()
    port = Portfolio("port_routing_test", assets, initCash=1_000_000)
    port.setDataSource(synth)
    acc1 = Account("acc_test_data_ref", assets=assets, initCash=1_000_000)
    acc1.setDataSource(synth)
    acc2 = Account("acc_test_data_ref", assets=assets, initCash=1_000_000)
    acc2.setDataSource(synth)
    acc1.addPortfolio(
        port)  # seg faults despite explicit cast from py::obj on c++ side
    acc1.addPortfolio("manually_added_port", assets, 1_000_000)
    acc1.addPortfolio(assets, 1_000_000)
    acc2.addPortfolio(port)
    ports = acc1.portfolios()
    assert (len(ports) == 4)
    port.setRequiredMargin(0.1)
    acc1.setRequiredMargin(0.1)
    acc1.setRequiredMargin("manually_added_port", 0.1)
    acc2.setRequiredMargin(0.1)
    port.handleTransaction('EURUSD', current_prices[0], 10_000, 0.)
    acc1.handleTransaction('manually_added_port', 'EURUSD', current_prices[0],
                           10_000., 0.)
    acc2.handleTransaction('EURUSD', current_prices[0], 10_000, 0.)
    manual_port = acc1.portfolioBook()['manually_added_port']
    try:
        assert port.equity == manual_port.equity == acc1.equity() / len(ports)
        assert port.cash == manual_port.cash == acc1.cash() - (
            (len(ports) - 1) * 1_000_000)
        assert port.borrowedMargin == manual_port.borrowedMargin == acc1.borrowedMargin(
        )
        del port  # no longer referenced
        acc1.portfolioBook()[
            'port_routing_test']  # test that acc contains own copy of port
        assert acc1.portfolioBook()['port_routing_test'].borrowedMargin == 0.
        acc1.handleTransaction('manually_added_port', 'EURUSD',
                               current_prices[0], 10_000., 0.)
        assert manual_port.cash > acc1.cash('manually_added_port'), \
            "portfolio from account api being referenced instead of copied"
        assert manual_port.borrowedMargin < acc1.borrowedMargin('manually_added_port'), \
            "portfolio from account api being referenced instead of copied"
        try:
            acc1.portfolio("port doesn't exist")  # THROWS INDEX ERROR
        except IndexError:
            pass
    except AssertionError:
        import ipdb
        ipdb.set_trace()


@pytest.mark.skip("util")
def compare_acc_transaction_ref(units, init_cash, required_margin):
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    data_source = Synth(config)
    assetIdx = 0
    acc = Account("port_accounting_test", assets, init_cash)
    acc.setDataSource(data_source)
    acc.setRequiredMargin(required_margin)

    current_prices = data_source.getData()
    # acc calc
    acc.handleTransaction(assetIdx, current_prices[assetIdx], units, 0.)
    # reference calc
    cash, borrowed_margin, equity = transaction(units=units,
                                                init_cash=init_cash,
                                                prices=current_prices,
                                                assetIdx=assetIdx,
                                                margin=required_margin)
    assert cash == acc.cash()
    assert borrowed_margin == acc.borrowedMargin()
    assert equity == acc.equity()


def test_acc_accounting_logic():
    compare_acc_transaction_ref(1000., 1_000_000, 1.)  # buy on cash
    compare_acc_transaction_ref(-1000., 1_000_000, 1.)  # sell on cash
    compare_acc_transaction_ref(1000., 1_000_000, .1)  # buy on margin (0.1)
    compare_acc_transaction_ref(-1000., 1_000_000, .1)  # sell on margin (0.1)


def test_broker_init():
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    acc = Account("port_accounting_test", assets, 1_000_000)
    broker1 = Broker("broker_init_test", assets, 1_000_000)
    broker2 = Broker(acc)
    broker3 = Broker(acc.portfolio())
    print("broker1 accounts", broker1.accounts())


def test_broker_data_ref():
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    synth = Synth()
    ref = synth.getData()
    acc = Account("acc_test_data_ref", assets=assets, initCash=1_000_000)
    acc.setDataSource(synth)
    assert isinstance(acc.currentPrices(), np.ndarray)
    assert_equal(synth.getData(), acc.currentPrices())


def compare_broker_transaction_ref(units, init_cash, required_margin):
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    data_source = Synth(config)
    assetIdx = 0
    broker = Broker("BrokerAcc", assets, init_cash)
    broker.setDataSource(data_source)
    broker.setRequiredMargin(required_margin)

    current_prices = data_source.getData()
    # acc calc
    broker.handleTransaction(assetIdx, units)
    # reference calc
    cash, borrowed_margin, equity = transaction(units=units,
                                                init_cash=init_cash,
                                                prices=current_prices,
                                                assetIdx=assetIdx,
                                                margin=required_margin)
    assert cash == broker.account().cash()
    assert borrowed_margin == broker.account().borrowedMargin()
    assert equity == broker.account().equity()


def test_broker_accounting_logic():
    compare_broker_transaction_ref(1000., 1_000_000, 1.)  # buy on cash
    compare_broker_transaction_ref(-1000., 1_000_000, 1.)  # sell on cash
    compare_broker_transaction_ref(1000., 1_000_000, .1)  # buy on margin (0.1)
    compare_broker_transaction_ref(-1000., 1_000_000,
                                   .1)  # sell on margin (0.1)


def test_broker_acc_port_routing():
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    synth = Synth()
    current_prices = synth.getData()
    port = Portfolio("port_routing_test", assets, initCash=1_000_000)
    broker = Broker("broker_acc", assets=assets, initCash=1_000_000)
    port.setDataSource(synth)
    broker.setDataSource(synth)
    broker.addPortfolio(port)
    broker.addPortfolio(Portfolio("manually_added_port", assets, 1_000_000))
    broker.setRequiredMargin("broker_acc", 0.1)
    broker.setRequiredMargin("broker_acc", "manually_added_port", 0.1)
    ports = broker.portfolios()
    assert (len(ports) == 3)
    price = current_prices[0]
    port.setRequiredMargin(0.1)
    port.handleTransaction('EURUSD', price, 10_000, 0.)
    broker.handleTransaction('broker_acc', 'EURUSD', 10_000.)
    broker.handleTransaction("broker_acc", "manually_added_port", 'EURUSD',
                             10_000)
    manual_port = broker.portfolioBook()['manually_added_port']
    manual_port_ref = broker.portfolio('manually_added_port')
    acc1 = broker.account()
    try:
        assert port.equity == manual_port.equity == broker.account().equity()/len(ports), \
            f'equity -> {port.equity} {manual_port.equity} {broker.account().equity()/len(ports)}'
        assert port.cash == manual_port.cash == acc1.cash() - (len(ports)-1)*(1_000_000) +\
            0.1*price*10_000
        assert port.borrowedMargin == manual_port.borrowedMargin == acc1.borrowedMargin(
        ) / 2
        del port  # no longer referenced
        assert broker.portfolioBook()['port_routing_test'].cash == 1_000_000.
        broker.handleTransaction('broker_acc', 'manually_added_port', 'EURUSD',
                                 10_000.)
        assert manual_port.cash > broker.account().cash('manually_added_port'), \
            "portfolio from broker.portfolioBook() being referenced instead of copied"
        assert manual_port.borrowedMargin < broker.account().borrowedMargin('manually_added_port'), \
            "portfolio from broker.portfolioBook() being referenced instead of copied"
        assert manual_port_ref.cash == broker.account().cash('manually_added_port'), \
            "portfolio from broker.portfolio(id) being copied instead of referenced"
        assert manual_port_ref.borrowedMargin == broker.account().borrowedMargin('manually_added_port'), \
            "portfolio from broker.portfolio(id) being copied instead of referenced"
        try:
            acc1.portfolio("port doesn't exist")  # THROWS INDEX ERROR
        except IndexError:
            pass
    except AssertionError as E:
        # import traceback; traceback.print_exc()
        # import ipdb; ipdb.set_trace();
        raise E


def test_broker_interface():
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    synth = Synth()
    current_prices = synth.getData()
    broker = Broker("broker_acc", assets=assets, initCash=1_000_000)
    broker.setDataSource(synth)
    brokerResponse = broker.handleAction([10_000, -10_000, 50_000, -200_000])


def test_broker_multi_trans():
    assets = Assets(['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHBTC'])
    synth = Synth()
    current_prices = synth.getData()
    broker = Broker("broker_acc", assets=assets, initCash=1_000_000)
    broker.setDataSource(synth)
    resp1 = broker.handleTransaction([10_000, -10_000, 50_000, -200_000])
    resp2 = broker.handleAction([-10_000, 10_000, -50_000, 200_000])


def test_successive_accounting1():
    """
    Round trip buy -> sell ->short _>buy back
    Buy 10_000 + 10_000 - 20_000 -20_000 + 10_000 + 10_000
    """
    synth = Synth()
    port = Portfolio("successive", assets, 1_000_000)
    port.setDataSource(synth)
    port.setRequiredMargin(0.1)
    prices = synth.getData()
    port.handleTransaction(0, prices[0], 10_000)  # BUY 10_000
    assert port.assetValue == prices[0] * 10_000
    port.handleTransaction(0, prices[0], 10_000)  # BUY 10_000
    assert port.cash == 1_000_000. - 0.1 * prices[0] * 20_000
    assert port.assetValue == prices[0] * 20_000
    assert port.usedMargin == 0.1 * prices[0] * 20_000
    assert port.borrowedMargin == 0.9 * prices[0] * 20_000
    assert port.borrowedAssetValue == 0.
    port.handleTransaction(0, prices[0], -20_000)  # SELL 20_000
    assert_allclose(port.cash, 1_000_000., rtol=1e-12)
    assert_allclose(port.assetValue, 0., rtol=1e-12)
    assert_allclose(port.usedMargin, 0., rtol=1e-12)
    assert_allclose(port.borrowedMargin, 0., rtol=1e-12)
    assert_allclose(port.borrowedAssetValue, 0., rtol=1e-12)
    port.handleTransaction(0, prices[0], -20_000)  # SELL 20_000
    assert_allclose(port.cash, 1_000_000 + prices[0] * 20_000., rtol=1e-12)
    assert_allclose(port.assetValue, prices[0] * -20_000, rtol=1e-12)
    assert_allclose(port.usedMargin, 0.1 * prices[0] * 20_000, rtol=1e-12)
    assert_allclose(port.borrowedMargin, 0.)
    assert_allclose(port.borrowedAssetValue, prices[0] * -20_000, rtol=1e-12)
    port.handleTransaction(0, prices[0], 10_000)  # BUY 10_000
    port.handleTransaction(0, prices[0], 10_000)  # BUY 10_000
    assert_allclose(port.cash, 1_000_000., rtol=1e-12)
    assert_allclose(port.assetValue, 0., rtol=1e-12)
    assert_allclose(port.usedMargin, 0., rtol=1e-12)
    assert_allclose(port.borrowedMargin, 0., rtol=1e-12)
    assert_allclose(port.borrowedAssetValue, 0., rtol=1e-12)


def test_successive_accounting2():
    """Short 10_000 then buy 20_000 and REVERSE position to 10_000"""
    synth = Synth()
    port = Portfolio("successive", assets, 1_000_000)
    port.setDataSource(synth)
    port.setRequiredMargin(0.1)
    prices = synth.getData()
    port.handleTransaction(0, prices[0], -10_000)  # SELL 10_000
    port.handleTransaction(0, prices[0], 20_000)  # BUY 20_000
    assert port.cash == 1_000_000. - 0.1 * prices[0] * 10_000
    assert port.assetValue == prices[0] * 10_000
    assert port.usedMargin == 0.1 * prices[0] * 10_000
    assert port.borrowedMargin == 0.9 * prices[0] * 10_000
    assert port.borrowedAssetValue == 0.


def test_successive_accounting3():
    """Long 10_000 then sell 20_000 and REVERSE position to -10_000"""
    synth = Synth()
    port = Portfolio("successive", assets, 1_000_000)
    port.setDataSource(synth)
    port.setRequiredMargin(0.1)
    prices = synth.getData()
    port.handleTransaction(0, prices[0], 10_000)  # BUY 10_000
    port.handleTransaction(0, prices[0], -20_000)  # BUY 10_000
    assert port.cash == 1_000_000. + prices[0] * 10_000
    assert port.assetValue == -prices[0] * 10_000
    assert port.usedMargin == 0.1 * prices[0] * 10_000
    assert port.borrowedMargin == 0.
    assert port.borrowedAssetValue == -prices[0] * 10_000


def test_multiasset_accounting():
    assets = Assets(['BTCUSD', 'ETHUSD', 'BTCETH', 'EURUSD'])
    synth = Synth()
    port = Portfolio("successive", assets, 1_000_000)
    port.setDataSource(synth)
    port.setRequiredMargin(0.1)
    prices = synth.getData()
    port.handleTransaction("BTCUSD", prices[0], 20_000)  # BUY 10_000
    assert port.cash == 1_000_000. - 0.1 * prices[0] * 20_000
    assert port.assetValue == prices[0] * 20_000
    assert port.usedMargin == 0.1 * prices[0] * 20_000
    assert port.borrowedMargin == 0.9 * prices[0] * 20_000
    assert port.borrowedAssetValue == 0.
    port.handleTransaction("EURUSD", prices[3], -20_000)  # SELL 20_000
    cash = 1_000_000. - (0.1 * prices[0] * 20_000) + (prices[3] * 20_000)
    balance = 1_000_000 - (0.1 * prices[0] * 20_000)
    assetValue = prices[0] * 20_000 + prices[3] * -20_000
    # usedMargin = 0.1*(prices[0]*20_000 + prices[3]*-20_000)
    usedMargin = 0.1 * prices[0] * 20_000 + 0.1 * prices[3] * 20_000
    borrowedMargin = 0.9 * prices[0] * 20_000
    borrowedAssetValue = prices[3] * -20_000
    try:
        assert_allclose(port.cash, cash, rtol=1e-12)
        assert_allclose(port.balance, balance, rtol=1e-12)
        assert_allclose(port.assetValue, assetValue, rtol=1e-12)
        assert_allclose(port.usedMargin, usedMargin, rtol=1e-12)
        assert_allclose(port.borrowedMargin, borrowedMargin, rtol=1e-12)
        assert_allclose(port.borrowedAssetValue,
                        borrowedAssetValue,
                        rtol=1e-12)
    except AssertionError as E:
        import traceback
        traceback.print_exc()
        import ipdb
        ipdb.set_trace()


def test_port_risk_handling():
    assets = Assets(['BTCUSD', 'ETHUSD', 'BTCETH', 'EURUSD'])
    synth = Synth()
    prices = synth.getData()
    prices[1] = 4
    price = prices[1]
    port = Portfolio("margin_call", assets, 1_000_000)
    port.setDataSource(synth)
    reqM = 0.1
    mainM = 1.
    port.setRequiredMargin(reqM)
    port.setMaintenanceMargin(mainM)
    port.handleTransaction("ETHUSD", price, 1_000_000)
    assert port.checkRisk("ETHUSD", (-1. + (port.balance + port.pnl) / reqM) /
                          price) == RiskInfo.green
    assert port.checkRisk("ETHUSD", (0. + (port.balance + port.pnl) / reqM) /
                          price) == RiskInfo.insuff_margin
    assert port.checkRisk("ETHUSD", (1. + (port.balance + port.pnl) / reqM) /
                          price) == RiskInfo.insuff_margin
    new_price = 3.71
    prices[1] = new_price
    assert port.checkRisk() == RiskInfo.green
    assert port.checkRisk("ETHUSD", 1_000_000 / price) == RiskInfo.green
    new_price = 3.69
    prices[1] = new_price
    assert port.checkRisk() == RiskInfo.margin_call
    assert port.checkRisk("ETHUSD", (-1. + (port.balance + port.pnl) / reqM) /
                          price) == RiskInfo.margin_call
    assert port.checkRisk("ETHUSD", 0.) == RiskInfo.margin_call
    loss = 1_000_000 * (price - new_price)
    equity = 1_000_000 - loss
    assert_allclose(-loss, port.pnl, rtol=1e-12)
    port.handleTransaction("ETHUSD", new_price, -1_000_000)
    cash = equity
    assert_allclose(equity, port.equity, rtol=1e-12)
    assert_allclose(cash, port.cash, rtol=1e-12)


def test_broker_risk_handling():
    assets = Assets(['BTCUSD', 'ETHUSD', 'BTCETH', 'EURUSD'])
    synth = Synth()
    prices = synth.getData()
    prices[1] = 4
    price = prices[1]
    broker = Broker("margin_call", assets, 1_000_000)
    broker.setDataSource(synth)
    reqM = 0.1
    mainM = 1.
    broker.setRequiredMargin(reqM)
    broker.setMaintenanceMargin(mainM)
    response = broker.handleTransaction("ETHUSD", 1_000_000)
    assert response.timestamp
    assert response.transactionPrice == 4
    assert response.transactionCost == 0.
    assert response.riskInfo == RiskInfo.green


def test_env_init():
    assets = Assets(['BTCUSD', 'ETHUSD', 'BTCETH', 'EURUSD'])
    env1 = Env("Synth", 1_000_000)
    env2 = Env("Synth", 1_000_000, config)
    print(env1.broker.portfolio())
    print(env1.portfolio)
    print(env2.broker.portfolio())
    print(env2.portfolio)


def test_env_interface():
    assets = Assets(['BTCUSD', 'ETHUSD', 'BTCETH', 'EURUSD'])
    env = Env("Synth", 1_000_000, config)
    srdi = env.step()
    state, reward, done, info = env.step([10000, 20000, -20000, -40000])


def test_env_accounting():
    assets = Assets(['BTCUSD', 'ETHUSD', 'BTCETH', 'EURUSD'])
    env = Env("Synth", 1_000_000, config)
    # port = Portfolio("PortRef", assets, 1_000_000)
    # port.setDataSource(env.dataSource)
    srdi1 = env.step()
    srdi2 = env.step(0, 10_000)  # BUY 10_000
    srdi3 = env.step(0, -20_000)  # SELL -20_000


def test_env_risk_handling():
    assets = Assets(['BTCUSD', 'ETHUSD', 'BTCETH', 'EURUSD'])
    env = Env("Synth", 1_000_000, config)
    srdi = env.step(0, -1 + 1_000_000 / env.currentPrices[0])  # BUY 10_000


if __name__ == "__main__":

    tests = [
        test_dataSource_init, test_dataSource_speed, test_buffer_referencing,
        test_assets_init, test_port_init, test_port_data_ref,
        test_port_accounting_logic, test_port_ledger, test_account_init,
        test_acc_data_ref, test_acc_port_routing, test_acc_accounting_logic,
        test_broker_init, test_broker_data_ref, test_broker_acc_port_routing,
        test_broker_accounting_logic, test_broker_interface,
        test_broker_multi_trans, test_successive_accounting1,
        test_successive_accounting2, test_successive_accounting3,
        test_multiasset_accounting, test_port_risk_handling,
        test_broker_risk_handling, test_env_init, test_env_interface,
        test_env_accounting, test_env_risk_handling
    ]
    completed = 0
    total = len(tests)
    try:
        for test in tests:
            test()
            completed += 1
    except Exception as E:
        raise E

    finally:
        print(f"{completed}/{total} TESTS COMPLETED")
