from random import random
import numpy as np
from ..utils import SARSD, State, ReplayBuffer
from ..fleet import DQN
from ..environments import Synth



def test(agent, env, nsteps=1000, eps=0., human_start=0):
    state = env.reset()
    eq = []
    returns = []
    prices = []
    positions = []
    cash = []
    margin = []
    for i in range(nsteps):
        if random() < eps or human_start > 0:
            action = env.action_space.sample()
            human_start -= 1
        else:
            action = agent(state)
        state, reward, done, info = env.step(action)
        if done:
            # print(info['Event'])
            break
        eq.append(env.equity)
        returns.append(reward)
        prices.append(env.current_prices)
        positions.append(env.portfolio)
        cash.append(env.cash)
        margin.append(env.available_margin)
    return {'eq': eq, 'returns': returns, 'prices': np.array(prices), 'positions': positions,
            'assets': env.assets, 'cash': cash, 'margin': margin}

