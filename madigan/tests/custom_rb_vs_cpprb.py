from timeit import timeit
import numpy as np

from madigan.utils import time_profile
from madigan.utils.data import SARSD, State
from madigan.utils.replay_buffer import ReplayBuffer as RBP # RB Python
from cpprb import ReplayBuffer as RBC # RB Cython

buffer_size = 1_000_000

obs_shape = 64
action_shape = 2
env_dict={"state_price": {"shape": obs_shape},
          "state_portfolio": {"shape": action_shape},
          "action": {"shape": action_shape},
          "next_state_price": {"shape": obs_shape},
          "next_state_portfolio": {"shape": action_shape},
          "reward": {},
          "done": {}}
          #"discounts": {}}

def make_data_dict(n, nstep=None):
    #if nstep is not None:
     #   discounts = 
    data={"state_price": np.random.randn(n, obs_shape).astype(np.float32),
          "state_portfolio": np.random.randn(n, action_shape).astype(np.float32),
          "action": np.random.randn(n, action_shape).astype(np.float32),
          "reward": np.random.randn(n).astype(np.float32).astype(np.float32),
          "next_state_price": np.random.randn(n, obs_shape).astype(np.float32),
          "next_state_portfolio": np.random.randn(n, action_shape).astype(np.float32),
          "done": np.random.binomial(1, 0.1, n)}
          #"discounts": np.random.randn(n).astype(np.float32)}
    return data

def make_data_tuple(n):
    return (np.random.randn(n, obs_shape).astype(np.float32),
            np.random.randn(n, action_shape).astype(np.float32), # port
            np.random.randn(n, action_shape).astype(np.float32), # actions for each asset
            np.random.randn(n).astype(np.float32),
            np.random.randn(n, obs_shape).astype(np.float32),
            np.random.randn(n, action_shape).astype(np.float32),
            np.random.binomial(1, 0.1, n))

def add_RBP_dict(rbp, data):
    for i in range(len(data['state_price'])):
        sarsd = SARSD(State(data['state_price'][i], data['state_portfolio'][i], None),
                      data['action'][i],
                      data['reward'][i],
                      State(data['next_state_price'][i], data['next_state_portfolio'][i], None),
                      data['done'][i])
        rbp.add(sarsd)

def add_RBP_tuple(rbp, data):
    for i in range(len(data[0])):
        sarsd = SARSD(State(data[0][i], data[1][i], None),
                      data[2][i],
                      data[3][i],
                      State(data[4][i], data[5][i], None),
                      data[6][i])
        rbp.add(sarsd)

def add_RBC_dict(rbc, data):
    rbc.add(**data)

def test_add_speed():
    # rbp = RBP(buffer_size, 1, 0.9)
    rbc = RBC(buffer_size, env_dict=env_dict)
    data_dict = make_data_dict(32)
    # data_tup = make_data_tuple(32)
    # time_py_dict = timeit(lambda: add_RBP_dict(rbp, data_dict), number=1000)
    # time_py_tuple = timeit(lambda: add_RBP_tuple(rbp, data_tup), number=1000)
    time_cy = timeit(lambda: add_RBC_dict(rbc, data_dict), number=1000)
    # print('timeit py add dict 32: ', time_py_dict)
    # print('timeit py add tuple 32: ', time_py_tuple)
    # print('timeit cy add dict 32: ', time_cy)
    # print('doing time_profile')
    # time_profile(1001, 0,
    #              add_RBP_dict = lambda: add_RBP_dict(rbp, data_dict),
    #              add_RBP_tuple = lambda: add_RBP_tuple(rbp, data_tup),
    #              add_RBC_dict = lambda: add_RBC_dict(rbc, data_dict))

def test_add_speed_nstep():
    # rbp = RBP(buffer_size, 30, 0.99)
    rbc = RBC(buffer_size, env_dict=env_dict, Nstep={"size": 30, "gamma": 0.99,
                                                     "rew": "reward",
                                                     "next": ["next_state_price",
                                                              "next_state_port"]})
    data_dict = make_data_dict(32)
    # data_tup = make_data_tuple(32)
    # time_py_dict = timeit(lambda: add_RBP_dict(rbp, data_dict), number=1000)
    # time_py_tuple = timeit(lambda: add_RBP_tuple(rbp, data_tup), number=1000)
    time_cy = timeit(lambda: add_RBC_dict(rbc, data_dict), number=1000)
    # print('timeit py add dict 32: ', time_py_dict)
    # print('timeit py add tuple 32: ', time_py_tuple)
    # print('timeit cy add dict 32: ', time_cy)
    # print('doing time profile')
    # time_profile(1000, 0,
    #              add_RBP_dict = lambda: add_RBP_dict(rbp, data_dict),
                #  add_RBP_tuple = lambda: add_RBP_tuple(rbp, data_tup),
                #  add_RBC_dict = lambda: add_RBC_dict(rbc, data_dict))

def test_sampling_speed():
    # rbp = RBP(buffer_size, 1, 0.99)
    rbc = RBC(buffer_size, env_dict=env_dict)
    data_dict = make_data_dict(32)
    # import ipdb; ipdb.set_trace()
    for i in range(1000):
        # add_RBP_dict(rbp, data_dict)
        add_RBC_dict(rbc, data_dict)

    rbc.on_episode_end()
    # time_py_sample = timeit(lambda: rbp.sample(32), number=1000)
    time_cy_sample = timeit(lambda: rbc.sample(32), number=1000)
    # print('py sample time: ', time_py_sample)
    print('cy sample time: ', time_cy_sample)
    # print('doing time profile')
    # time_profile(1000, 0, py_sample=lambda: rbp.sample(32),
    #              cy_sample=lambda: rbc.sample(32))


def test_sarsd_same(sample_py, sample_cy):
    for i, (a, b) in enumerate(zip([sample_py.state.price, sample_py.state.portfolio,
                    sample_py.action, sample_py.reward,
                    sample_py.next_state.price, sample_py.next_state.portfolio,
                    sample_py.done],
                    [sample_cy['state_price'], sample_cy['state_portfolio'],
                    sample_cy['action'], sample_cy['reward'],
                    sample_cy['next_state_price'], sample_cy['next_state_portfolio'],
                    sample_cy['done']])):
        try:
            np.testing.assert_allclose(a, b.squeeze(), atol=1e-6)
        except AssertionError as E:
            print(E)
            print('data num: ', i)
            print('shapes: ', a.shape, b.shape)

def test_1step_logic():
    rbp = RBP(buffer_size, 1, 0.9)
    rbc = RBC(buffer_size, env_dict=env_dict)
    data = make_data_dict(32)
    add_RBP_dict(rbp, data)
    add_RBC_dict(rbc, data)
    # rbc.on_episode_end()
    sample_py = rbp.get_full()
    sample_cy = rbc.get_all_transitions()
    test_sarsd_same(sample_py, sample_cy)

def test_nstep_logic():
    rbp = RBP(buffer_size, 3, 0.99)
    rbc = RBC(buffer_size, env_dict=env_dict, Nstep={"size": 3, "gamma": 0.99,
                                                     "rew": "reward",
                                                     "next": ["next_state_price",
                                                              "next_state_port"]})
    data = make_data_dict(32)
    add_RBP_dict(rbp, data)
    add_RBC_dict(rbc, data)
    sample_py = rbp.get_full()
    sample_cy = rbc.get_all_transitions()
    # import ipdb; ipdb.set_trace()
    test_sarsd_same(sample_py, sample_cy)

if __name__ == "__main__":
    print("*"*80)
    print("Single Step Adding")
    test_add_speed()
    print("*"*80)
    print("NStep Adding")
    test_add_speed_nstep()
    print("*"*80)
    print("Sampling")
    test_sampling_speed()
    print("*"*80)
    print("Test 1Step Logic")
    test_1step_logic()
    print("*"*80)
    print("Test nStep Logic")
    test_nstep_logic()


