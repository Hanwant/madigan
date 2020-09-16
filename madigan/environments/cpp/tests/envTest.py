import numpy as np
from numpy import isclose
from numpy.testing import assert_equal
from madigan.environments.cpp import Portfolio, Synth, Account, Asset, Assets, Broker#, Env

def test_dataSource_init():
    synth = Synth()
    synth_ = Synth([1., 0.3, 2., 0.5], [2., 2.1, 2.2, 2.3],
                  [1., 1.2, 1.3, 1.], [0., 1., 2., 1.],
                  dx=0.01)
    prices = synth.getData()
    prices_ = synth_.getData()
    assert isinstance(prices, np.ndarray)
    assert isinstance(prices_, np.ndarray)
    assert_equal(prices, prices_)

def test_buffer_referencing():
    synth = Synth()
    dat = synth.getData()
    arr_default = np.array(dat)
    arr_ref = np.array(dat, copy=False)
    arr_copy = np.array(dat, copy=True)
    dat[0] = 33333
    assert arr_ref[0] == dat[0], "pybind is not returning by reference"
    assert arr_copy[0] != dat[0]
    if arr_default[0] == dat[0]:
        print("default ndarray constructor references buffer without copy")
    else:

        print("default ndarray constructor copies buffer ")

def test_assets_init():
    asset1 = Asset("EURUSD")
    asset2 = Asset("GBPUSD")
    assets = Assets(['EURUSD', 'GBPUSD'])
    assets_ = Assets([asset1, asset2])
    assert all((ass1.code==ass2.code for ass1, ass2 in zip(assets, assets_)))

def test_port_init():
    assets = Assets(['EURUSD', 'GBPUSD'])
    port1 = Portfolio()
    port2 = Portfolio("port_1", assets=assets, initCash=1_000_000)

def test_account_init():
    assets = Assets(['EURUSD', 'GBPUSD'])
    port = Portfolio("REF PORT", assets, 1_000_000)
    account1 = Account()
    account2 = Account("coinbase_acc", assets=assets, initCash=1_000_000)
    account3 = Account(assets=assets, initCash=1_000_000)
    print("doing port init")
    account4 = Account(port)
    del port # ACCOUNT DOES NOT REFER TO PORT PASSED IN CONSTRUCTOR
    print("account1 port", account1.portfolios())
    print("account2 port", account2.portfolios())
    print("account3 port", account3.portfolios())
    print("account4 port", account4.portfolios())


def test_broker_init():
    assets = Assets(['EURUSD', 'GBPUSD'])
    account = Account("acc", assets, 1_000_000)
    broker1 = Broker()
    broker2 = Broker("broker_cons_acc", assets, 1_000_000)
    broker3 = Broker(account)
    broker4 = Broker(account.portfolio())

def test_env_init():
    pass

def test_accounting_logic():
    pass

def test_env_interface():
    pass

def test_env_logic():
    pass

if __name__ == "__main__":
    from madigan.utils import time_profile
    from madigan.environments import Synth as Synth_py

    test_dataSource_init()
    test_buffer_referencing()
    test_assets_init()
    test_port_init()
    test_account_init()
    test_broker_init()
    test_env_init()
    test_accounting_logic()
    test_env_interface()
    test_env_logic()

    synth = Synth()
    dat = synth.getData()
    arr = np.array(dat)
    arr_ref = np.array(dat, copy=False)
    arr_copy = np.array(dat, copy=True)

    synth_py = iter(Synth_py(discrete_actions=True)._data_stream)
    time_profile(1000, 1, py_synth=lambda: next(synth_py), cppsynth=lambda: synth.getData())
