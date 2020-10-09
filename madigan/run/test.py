from pathlib import Path
from random import random
import numpy as np
import matplotlib.pyplot as plt
from ..utils.plotting import make_grid
from ..utils.preprocessor import make_preprocessor
from ..fleet import make_agent
from ..environments import make_env


def test_loop(agent, env, preprocessor, eps=0., random_starts=0):
    state = env.reset()
    preprocessor.stream_state(state)
    # try:
    done = False
    while True:
        if len(preprocessor) >= agent.min_tf: # if enough time series points have been accumulated
            if random() < eps or random_starts > 0:
                action = agent.action_space.sample()
                random_starts -= 1
            else:
                action = agent(preprocessor.current_data()) # preprocessed data fed into agent
                state, reward, done, info = env.step(action)
                # if env.cash < 0. or env.availableMargin < 0.:
                #     import ipdb; ipdb.set_trace()
                preprocessor.stream_state(state)
            if done:
                reward = -1.
                yield state, reward, done, info
                break
            yield state, reward, done, info
        else:
            state, reward, done, info = env.step()
            preprocessor.stream_state(state) # no action, accumulate data
            yield None
    # except:
    #     import traceback; traceback.print_exc()
    #     import ipdb; ipdb.set_trace()

def test(agent, env, preprocessor, nsteps=1000, eps=0., random_starts=0,
         verbose=False):
    loop = test_loop(agent, env, preprocessor, eps=eps, random_starts=random_starts)
    equity = []
    returns = []
    prices = []
    positions = []
    cash = []
    margin = []
    actions = []
    done=False
    try:
        for i in range(nsteps):
            metrics = next(loop)
            if metrics is not None:
                state, reward, done, info = metrics
                equity.append(env.equity)
                returns.append(reward)
                prices.append(np.array(env.currentPrices, copy=True))
                states.append(state)
                # positions.append(np.array(env.ledger, copy=True))
                positions.append(env.ledgerNormed)
                cash.append(env.cash)
                margin.append(env.availableMargin)
                actions.append(info.brokerResponse.transactionUnits)
    except StopIteration as SE:
        if verbose:
            if done:
                # import ipdb; ipdb.set_trace()
                print('transaction risk: ', info.brokerResponse.riskInfo)
                print('margin call: ', info.brokerResponse.marginCall)
                # print("pnl", env.pnlPositions)
                # print("pnl total", env.pnl)
                # print("position values", env.positionValues)
                # print("total asset valu", env.assetValue)
            print(f'Stopped at {i} steps')
    return {'equity': equity, 'returns': returns, 'prices': np.array(prices), 'positions': positions,
            'assets': env.assets, 'cash': cash, 'margin': margin, 'actions': actions}


def get_int_from_user():
    try:
        user_input = input()
        if user_input == "q":
            raise StopIteration("User signalled exit")
        elif user_input == '':
            return 0
        user_input = int(user_input)
        return user_input
    except ValueError as E:
        print(E)
        return get_int_from_user()

def test_loop_manual(env, preprocessor):
    state = env.reset()
    preprocessor.stream_state(state)
    # try:
    done = False
    nactions = len(env.assets)
    print(f"{nactions} required for each action")
    print(f"at each iteration, enter {nactions} successive actions (I.e transaction amounts)")
    print(f"Enter 'q' to quit")
    try:
        while True:
            if len(preprocessor) >= preprocessor.min_tf: # if enough time series points have been accumulated
                print("state price: ", state.price)
                print("state portfolio: ", state.portfolio)
                print("state timestamp: ", state.timestamp)
                print("state preprocessed (last 10 steps): ", np.array(preprocessor.current_data().price[-10:]))
                print('eq: ', env.equity)
                print('cash: ', env.cash)
                print('port ledger: ', env.ledger)
                print('pnl: ', env.pnlPositions)
                print('balance: ', env.portfolio.balance)
                print('available margin: ', env.availableMargin)
                print('used margin: ', env.usedMargin)
                print('borrowedAssetValue: ', env.borrowedAssetValue)
                actions = []
                for i in range(nactions):
                    action = get_int_from_user()
                    actions.append(action)
                state, reward, done, info = env.step(actions)
                preprocessor.stream_state(state)
                if done:
                    reward = -1.
                print('brokerResponse: ', info.brokerResponse)
                print('reward: ', reward)
                if done:
                    yield reward, done, info
                    break
                yield reward, done, info
            else:
                state, reward, done, info = env.step()
                preprocessor.stream_state(state) # no action, accumulate data
                yield None
    except StopIteration as SE:
        return # propagates stopiteration to wrapper

def test_manual(env, preprocessor, nsteps=1000, verbose=False):
    loop = test_loop_manual(env, preprocessor)
    equity = []
    returns = []
    prices = []
    positions = []
    cash = []
    margin = []
    actions = []
    done=False
    try:
        for i in range(nsteps):
            metrics = next(loop)
            if metrics is not None:
                reward, done, info = metrics
                equity.append(env.equity)
                returns.append(reward)
                prices.append(np.array(env.currentPrices, copy=True))
                # positions.append(np.array(env.ledger, copy=True))
                positions.append(env.ledgerNormed)
                cash.append(env.cash)
                margin.append(env.availableMargin)
                actions.append(info.action)
    except StopIteration as SE:
        if verbose:
            if done:
                print(info.brokerResponse.riskInfo)
            print(f'Stopped at {i} steps')
    return {'equity': equity, 'returns': returns, 'prices': np.array(prices), 'positions': positions,
            'assets': env.assets, 'cash': cash, 'margin': margin}
