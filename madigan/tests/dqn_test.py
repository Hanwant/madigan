import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
from madigan.environments import make_env
from madigan.environments.cpp import Synth
from madigan.fleet import make_agent
from madigan.fleet.dqn import DQN
from madigan.utils import make_config, State, SARSD, batchify_sarsd, ReplayBuffer
from madigan.utils.preprocessor import make_preprocessor


@pytest.mark.skip("util")
def make_random_data(config, sample_size=1):
    price_shape = (sample_size, config['min_tf'], config['n_assets'])
    port_shape = (sample_size, config['n_assets'])
    price = np.random.randn(*price_shape)
    port = np.random.randn(*port_shape)
    state = State(price, port, 0)
    action = np.random.randint(0, config.agent_config.action_atoms, sample_size)
    reward = np.random.randn(sample_size)
    price = np.random.randn(*price_shape)
    port = np.random.randn(*port_shape)
    next_state = State(price, port, 0)
    done = np.random.binomial(1, 0.5, sample_size)
    return SARSD(state, action, reward, next_state, done)



def test_conv_forward_pass():
    config = make_config(
        experiment_id="dqn_test",
        basepath="/media/hemu/Data/Markets/farm",
        overwrite_exp=False,
        nsteps=1_000_000,
        assets=["sine1"],
        preprocessor_type="WindowedStacker",
        window_length=64,
        discrete_actions=True,
        discrete_action_atoms=11,
        model_class="ConvModel",
        lr=1e-3,
        double_dqn=True,
        rb_size=100_000,
        train_freq=4,
        test_freq=32000,
        lot_unit_value=1_000,
        generator_params={
            'freq':[1.],
            'mu':[2.],
            'amp':[1.],
            'phase':[0.],
            'dX':0.01}
    )
    dqn = DQN(config, name="test")
    n_feats = config['agent_config']['model_config']['n_feats']

    # price_shape is (bs, tf, feats) - tf and feats are switched inside model for convolution
    price_shape = (1, config['min_tf'], config['n_assets'])
    port_shape = (1, config['n_assets'])

    price = np.random.randn(*price_shape)
    port = np.random.randn(*port_shape)
    state = State(price, port, 0)
    action = dqn(state, target=True)
    # import ipdb; ipdb.set_trace()
    assert tuple(action.shape) == (config.n_assets, )


def test_conv_random_batch_overfit():
    config = make_config(
        experiment_id="dqn_test",
        basepath="/media/hemu/Data/Markets/farm",
        overwrite_exp=False,
        nsteps=1_000_000,
        assets=["sine1"],
        preprocessor_type="WindowedStacker",
        window_length=64,
        discrete_actions=True,
        discrete_action_atoms=11,
        model_class="ConvModel",
        lr=1e-3,
        double_dqn=True,
        rb_size=100_000,
        train_freq=4,
        test_freq=32000,
        lot_unit_value=1_000,
        generator_params={
            'freq':[1.],
            'mu':[2.],
            'amp':[1.],
            'phase':[0.],
            'dX':0.01}
    )
    agent = make_agent(config)
    device = 'cuda'
    # env = Synth(**config)
    # state = env.reset()
    # action = env.action_space.sample()
    # next_state, reward, done, info = env.step(action)
    # sarsd = batchify_sarsd(SARSD(state, action, reward, next_state, done))
    sarsd = make_random_data(config, sample_size=32)
    qvals = []
    for i in range(2000):
        # metrics = train_step(agent, sarsd, device)
        metrics = agent.train_step(sarsd)
        qvals.append([metrics['G_t'], metrics['Q_t']])
        if i % 100 == 0:
            print('step: ', i, 'loss: ', f"{metrics['loss']:.2e}", 'td_error: ', '{:.2e}'.format(metrics['td_error']))
    qvals_match = [torch.allclose(q[0], q[1]) for q in qvals]
    assert sum(qvals_match) > 1

def test_conv_synth_batch_overfit():
    config = make_config(
        experiment_id="dqn_test",
        basepath="/media/hemu/Data/Markets/farm",
        overwrite_exp=False,
        nsteps=1_000_000,
        assets=["sine1"],
        preprocessor_type="WindowedStacker",
        window_length=64,
        discrete_actions=True,
        discrete_action_atoms=11,
        model_class="ConvModel",
        lr=1e-3,
        double_dqn=True,
        rb_size=100_000,
        train_freq=4,
        test_freq=32000,
        lot_unit_value=1_000,
        generator_params={
            'freq':[1.],
            'mu':[2.],
            'amp':[1.],
            'phase':[0.],
            'dX':0.01}
    )
    agent = DQN(config, name="test")
    env = make_env(config)
    preprocessor = make_preprocessor(config)
    rb = ReplayBuffer(config.rb_size)

    window_length = config.preprocessor_config.window_length

    preprocessor.initialize_history(env)
    state = preprocessor.current_data()
    episode_metrics=[]
    for i in range(window_length):
        # action = agent(state)
        action = np.random.choice([-config.lot_unit_value, 0, config.lot_unit_value],
                                  p=[0.05,                0.9,                 0.05],
                                  size=len(config.assets))
        _next_state, reward, done, info = env.step(action)
        preprocessor.stream_state(_next_state)
        next_state = preprocessor.current_data()
        rb.add(SARSD(state, action, reward, next_state, done))
        if done:
            env.reset()
            preprocessor.initialize_history(env)
            state = preprocessor.current_data()
        else:
            state = next_state
        episode_metrics.append({'actions': action,
                                'prices': state.price[-1]
        episode_metrics['states'].append(action)
        episode_metrics['rewards'].append(action)
        episode_metrics['dones'].append(action)
        episode_metrics['infos'].append(action)

    train_metrics = {'qvals': [], 'losses': []}
    train_metrics = []
    for i in range(2000):
        sarsd = rb.sample(window_length)
        metrics = agent.train_step(sarsd)
        train_metrics.append(metrics)
        if i % 100 == 0:
            print('step: ', i, 'loss: ', f"{train_metrics['loss']:.2e}",
                  f"Q_t: {train_metrics[i]['Q_t']:.2e}",
                  f"G_t: {train_metrics[i]['G_t']:.2e}",
                  'td_error: ', '{:.2e}'.format(metrics['td_error']))
    qvals_match = [torch.allclose(q[0], q[1]) for q in qvals]

    episode = 
    fig, ax = plt.subplots(2, 2)
    _ax = ax.flatten()
    for i in enumerate()
    _ax[0].plot()

    # assert sum(qvals_match) > 1

if __name__=="__main__":
    test_conv_forward_pass()
    # test_conv_random_batch_overfit()
    test_conv_synth_batch_overfit()
