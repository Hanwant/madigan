from random import random
import numpy as np

def test_loop(agent, env, eps=0., random_starts=0):
    state = env.reset()
    while True:
        if random() < eps or random_starts > 0:
            action = env.action_space.sample()
            random_starts  -= 1
        else:
            action = agent(state)
        state, reward, done, info = env.step(action)
        if done:
            # print(info['Event'])
            break
        yield reward, done, info

def test(agent, env, nsteps=1000, eps=0., random_starts=0,
         verbose=False):
    loop = test_loop(agent, env, eps=eps, random_starts=random_starts)
    equity = []
    returns = []
    prices = []
    positions = []
    cash = []
    margin = []
    try:
        for i in range(nsteps):
            reward, done, info = next(loop)
            equity.append(env.equity)
            returns.append(reward)
            prices.append(env.current_prices)
            positions.append(env.portfolio)
            cash.append(env.cash)
            margin.append(env.available_margin)
    except StopIteration:
        if verbose:
            if done:
                print(info['Event'])
                print(f'Stopped at {nsteps} steps')
            else:
                print(f'Stopped at {nsteps} steps')
    return {'equity': equity, 'returns': returns, 'prices': np.array(prices), 'positions': positions,
            'assets': env.assets, 'cash': cash, 'margin': margin}

