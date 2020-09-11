# from env import Broker, Env, Portfolio, Account, Synth
from env import Portfolio, Synth

def test_dataSource_init():
    synth = Synth()
    assert synth.getData() is not None

def test_port_init():
    port = Portfolio(nassets=4, initCash=1_000_000)

def test_account_init():
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
