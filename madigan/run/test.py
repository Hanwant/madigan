from pathlib import Path
from random import random
import numpy as np
import matplotlib.pyplot as plt
from ..utils.metrics import list_2_dict
from ..utils.plotting import make_grid
from ..utils.preprocessor import make_preprocessor
from ..modelling import make_agent
from ..environments import make_env

def get_test_loop(agent, env, preprocessor, nsteps=1000, reset=False, eps=0.,
                  random_starts=0, boltzmann=False, boltzmann_temp=1., verbose=False):

    config = agent.config
    agent_type = config.agent_type
    if agent_type in ("DQN", "LQN"):
        test_loop = test_loop_dqn(agent, env, preprocessor, eps=eps,
                                  random_starts=random_starts, reset=reset,
                                  boltzmann=boltzmann, boltzmann_temp=boltzmann_temp)
        return test_loop
    if agent_type in ("A2C"):
        test_loop = test_loop_actor_critic(agent, env, preprocessor, nsteps=nsteps,
                                           reset=reset, eps=eps,
                                           random_starts=random_starts)
        return test_loop
    if agent_type in ("Actor"):
        test_loop = test_loop_actor(agent, env, preprocessor, nsteps=nsteps,
                                           reset=reset, eps=eps,
                                           random_starts=random_starts)
        return test_loop
    raise NotImplementedError(f"test loop for {config.agent_type} is not implemented")

def test(agent, env, preprocessor, nsteps=1000, reset=False, eps=0., random_starts=0,
         boltzmann=False, boltzmann_temp=1., verbose=False):
    loop = get_test_loop(agent, env, preprocessor, eps=eps, random_starts=random_starts,
                         reset=reset, boltzmann=boltzmann, boltzmann_temp=boltzmann_temp)
    done=False
    metrics = []
    try:
        for i in range(nsteps):
            _metrics = next(loop)
            if _metrics is not None:
                _metrics['equity'] = env.equity
                _metrics['prices'] = np.array(env.currentPrices, copy=True)
                _metrics['reward'] = _metrics['reward']
                _metrics['positions'] = env.ledgerNormed
                _metrics['cash'] = env.cash
                _metrics['margin'] = env.availableMargin
                _metrics['transactions'] = _metrics['info'].brokerResponse.transactionUnits
            metrics.append(_metrics)
    except StopIteration as SE:
        if verbose:
            done = _metrics['done']
            info = _metrics['info']
            if done:
                print('transaction risk: ', info.brokerResponse.riskInfo)
                print('margin call: ', info.brokerResponse.marginCall)
                # print("broker checkrisk", env.portfolio.checkRisk())
                # print("pnl", env.pnlPositions)
                # print("pnl total", env.pnl)
                # print("position values", env.positionValues)
                # print("total asset valu", env.assetValue)
            print(f'Stopped at {i} steps')
    return list_2_dict(metrics)

def test_loop_dqn(agent, env, preprocessor, eps=0., random_starts=0, reset=False,
                  boltzmann=False, boltzmann_temp=1.):
    # try:
    done = False
    if reset:
        env.reset()
    preprocessor.initialize_history(env)
    state = preprocessor.current_data()
    example_qvals = agent.get_qvals(preprocessor.current_data())[0].cpu().numpy()
    default_qvals = np.zeros_like(example_qvals)
    while True:
        if random() < eps or random_starts > 0:
            action = agent.action_space.sample()
            random_starts -= 1
            qvals = default_qvals
        else:
            current_data = preprocessor.current_data()
            action = agent(current_data) # preprocessed data fed into agent
            qvals = agent.get_qvals(current_data)[0].cpu().numpy()
        state, reward, done, info = env.step(action)
        preprocessor.stream_state(state)
        if done:
            # reward = -1.
            yield {'state': state, 'reward': reward, 'done': done,
                   'info': info, 'qvals': qvals}
            break
        yield {'state': state, 'reward': reward, 'done': done,
               'info': info, 'qvals': qvals}

def test_loop_actor_critic(agent, env, preprocessor, eps=0., random_starts=0,
                           reset=False, **params):
    # try:
    done = False
    if reset:
        env.reset()
    preprocessor.initialize_history(env)
    state = preprocessor.current_data()
    example_state_val = agent.get_state_value(preprocessor.current_data()
                                             )[0].detach().cpu().numpy()
    default_state_val = np.zeros_like(example_state_val)
    example_action_probs = agent.get_policy(preprocessor.current_data()
                                             ).probs.detach().cpu().numpy()
    default_probs = np.zeros_like(example_action_probs)
    while True:
        if random() < eps or random_starts > 0:
            action = agent.action_space.sample()
            random_starts -= 1
            state_val = default_state_val
            probs = default_probs
        else:
            state = preprocessor.current_data()
            policy = agent.get_policy(state)
            probs = policy.probs
            action = agent(state) # preprocessed data fed into agent
            state_val = agent.get_state_value(state
                                              )[0].detach().cpu().numpy()
        state, reward, done, info = env.step(action)
        preprocessor.stream_state(state)
        if done:
            # reward = -1.
            yield {'state': state, 'reward': reward, 'done': done,
                   'info': info, 'state_val': state_val,
                   'action_probs': probs.detach().cpu().numpy()}
            break
        yield {'state': state, 'reward': reward, 'done': done,
               'info': info, 'state_val': state_val,
               'action_probs': probs.detach().cpu().numpy()}
def test_loop_actor(agent, env, preprocessor, eps=0., random_starts=0,
                           reset=False, **params):
    # try:
    done = False
    if reset:
        env.reset()
    preprocessor.initialize_history(env)
    state = preprocessor.current_data()
    example_action_probs = agent.get_policy(preprocessor.current_data()
                                             ).probs.detach().cpu().numpy()
    default_probs = np.zeros_like(example_action_probs)
    while True:
        if random() < eps or random_starts > 0:
            action = agent.action_space.sample()
            random_starts -= 1
            probs = default_probs
        else:
            state = preprocessor.current_data()
            policy = agent.get_policy(state)
            probs = policy.probs
            action = agent(state) # preprocessed data fed into agent
        state, reward, done, info = env.step(action)
        preprocessor.stream_state(state)
        if done:
            # reward = -1.
            yield {'state': state, 'reward': reward,
                   'done': done, 'info': info,
                   'action_probs': probs.detach().cpu().numpy()}
            break
        yield {'state': state, 'reward': reward,
                'done': done, 'info': info,
                'action_probs': probs.detach().cpu().numpy()}

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
