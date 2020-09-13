import numpy as np
from madigan.environments.cpp import Portfolio, Synth, Account

def test_dataSource_init():
    synth = Synth()
    # synth = Synth([1., 0.3, 2., 0.5],[2., 2.1, 2.2, 2.3],
    #               [1., 1.2, 1.3, 1.], [0., 1., 2., 1.],
    #               dx=0.01)
    assert synth.getData() is not None

def test_buffer_referencing():
    synth = Synth()
    dat = synth.getData()
    arr_default = np.array(dat)
    arr_ref = np.array(dat, copy=False)
    arr_copy = np.array(dat, copy=True)

def test_port_init():
    port = Portfolio("port_1", nAssets=4, initCash=1_000_000)

def test_account_init():
    account = Account("coinbase_acc", nAssets=4, initCash=1_000_000)
    # port = Portfolio(nAssets=3, initCash=1_000_000)
    # account.addPortfolio(port)
    pass


def test_broker_init():
    pass

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
