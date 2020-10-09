import numpy as np
from madigan.environments.cpp import Env, State, RiskInfo, EnvInfoMulti, Assets
from madigan.utils import time_profile


class EnvWrap(Env):
    def step(self, units: np.ndarray = None):
        """
        Exact same logic as Env.step() and Env.step(units)
        implemented in cpp - overriding from python allows more control
        """
        if units is None: # no transaction
            prevEq = self.portfolio.equity
            newPrices = self.dataSource.getData()
            newEq= self.portfolio.equity
            reward = (newEq-prevEq) / prevEq
            risk = self.portfolio.checkRisk()
            done = False if risk == RiskInfo.green else True
            return (State(newPrices, self.ledgerNormed,
                          self.dataSource.currentTime),
                    reward, done, EnvInfoMulti())
        else:
            if not isinstance(units, np.ndarray):
                raise TypeError("units must be an np array")
            prevEq = self.portfolio.equity
            broker_response_multi = self.broker.handleTransaction(units)
            newPrices = self.dataSource.getData()
            newEq= self.portfolio.equity
            reward = (newEq-prevEq) / prevEq
            risk = self.portfolio.checkRisk()
            done = False if risk == RiskInfo.green else True
            for _risk in broker_response_multi.riskInfo:
                if _risk != RiskInfo.green:
                    done = True
            return (State(newPrices, self.ledgerNormed,
                          self.dataSource.currentTime),
                    reward, done, EnvInfoMulti(broker_response_multi))

def test_wrapper_logic():
    assets = Assets(["sine1", "sine2", "sine3", "sine4"])
    env_c = Env("Synth", assets, 1_000_000)
    env_py = EnvWrap("Synth", assets, 1_000_000)
    for i in range(100):
        action_units = 1000*np.random.randn(4)
        srdi_c = env_c.step(action_units)
        srdi_py = env_py.step(action_units)
        assert srdi_c[1] == srdi_py[1], "rewards are not same"
        assert srdi_c[2] == srdi_py[2], "'done' is not the same"
        assert srdi_c[3].brokerResponse.riskInfo == srdi_py[3].brokerResponse.riskInfo,\
            "risk Info is not the same"

def test_wrapper_speed():
    assets = Assets(["sine1", "sine2", "sine3", "sine4"])
    env_c = Env("Synth", assets, 1_000_000)
    env_py = EnvWrap("Synth", assets, 1_000_000)

    def time_env_c_default():
        srdi = env_c.step()
    def time_env_py_default():
        srdi = env_py.step()

    # contains the time taken to make action_units
    def time_env_c_action():
        action_units = 1000*np.random.randn(4)
        srdi = env_c.step(action_units)
    def time_env_py_action():
        action_units = 1000*np.random.randn(4)
        srdi = env_py.step(action_units)

    time_profile(10000, 0,
                 env_c__default=time_env_c_default,
                 env_py_default=time_env_py_default,
                 env_c__action=time_env_c_action,
                 env_py_action=time_env_py_action,
                 )


if __name__=="__main__":
    test_wrapper_logic()
    test_wrapper_speed()

