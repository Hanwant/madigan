from functools import partial
from random import random
import numpy as np
import pytest
from madigan.utils.config import make_config
from madigan.environments import make_env
from madigan.fleet import make_agent
from madigan.run.test import test, test_loop
from madigan.utils import time_profile

@pytest.mark.skip("function to be used inside test")
def test_normal_loop(agent, env, nsteps=1000, eps=0., random_starts=0):
    state = env.reset()
    equity = []
    returns = []
    prices = []
    positions = []
    cash = []
    margin = []
    for i in range(nsteps):
        if random() < eps or random_starts > 0:
            action = env.action_space.sample()
            random_starts -= 1
        else:
            action = agent(state)
        state, reward, done, info = env.step(action)
        if done:
            # print(info['Event'])
            break
        equity.append(env.equity)
        returns.append(reward)
        prices.append(env.current_prices)
        positions.append(env.portfolio)
        cash.append(env.cash)
        margin.append(env.available_margin)
    return {'equity': equity, 'returns': returns, 'prices': np.array(prices), 'positions': positions,
            'assets': env.assets, 'cash': cash, 'margin': margin}


def test_generator_vs_normal_loop():
    config = make_config(discrete_actions=True, discrete_action_atoms=11, env_type="Synth",
                        model_class="ConvModel", lr=1e-3, double_dqn=True, lot_unit_value=1000)
    env = make_env(config)
    agent = make_agent(config)
    nsteps = 1000
    device = 'cuda'
    metrics = test(agent, env, nsteps=nsteps)
    time_profile(3, 0, generator=partial(test, agent, env, nsteps=nsteps),
                 normal_loop=partial(test_normal_loop, agent, env, nsteps=nsteps))


if __name__ == "__main__":
    test_generator_vs_normal_loop()
