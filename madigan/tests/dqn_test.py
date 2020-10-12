import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
from madigan.run.test import test
from madigan.environments import make_env
from madigan.environments.cpp import Synth
from madigan.fleet import make_agent
from madigan.fleet.dqn import DQN
from madigan.utils import make_config, State, SARSD, batchify_sarsd, ReplayBuffer
from madigan.utils.preprocessor import make_preprocessor
from madigan.utils import list_2_dict, reduce_test_metrics, reduce_train_metrics
from madigan.utils.plotting import plot_test_metrics, plot_train_metrics, plot_sarsd


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
        discrete_action_atoms=3,
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
        discrete_action_atoms=3,
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
            'dX':0.01,
            "noise": 0.01}
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
        test_steps=1_000,
        nsteps=1_000_000,
        assets=["sine1"],
        preprocessor_type="WindowedStacker",
        window_length=64,
        discrete_actions=True,
        discrete_action_atoms=3,
        model_class="ConvModel",
        lr=1e-3,
        double_dqn=True,
        target_update_freq=1000,
        rb_size=10_000,
        min_rb_size=128,
        lot_unit_value=1_000,
        generator_params={
            'freq':[1.],
            'mu':[2.],
            'amp':[1.],
            'phase':[0.],
            'dX':0.01,
            "noise": 0.1}
    )
    agent = DQN(config, name="test")
    env = make_env(config)
    preprocessor = make_preprocessor(config)
    rb = ReplayBuffer(config.rb_size)

    window_length = config.preprocessor_config.window_length
    action_choices = np.array([-config.lot_unit_value, 0, config.lot_unit_value],
                              dtype=np.float64)
    target_update_freq = config.target_update_freq

    pre = list_2_dict(test(agent, env, preprocessor, nsteps=config.test_steps,
                           eps=0., random_starts=0, boltzmann=False, boltzmann_temp=1.,
                           verbose=True))
    env.reset()
    preprocessor.initialize_history(env)
    state = preprocessor.current_data()
    train_metrics = []
    eps = 1.
    eps_decay = 0.99999
    for i in range(20000):
        eps *= eps_decay
        if np.random.random() < eps:
            action = np.random.choice(action_choices, p=[0.1, 0.8, 0.1],
                                      size=len(config.assets))
        else:
            qvals = agent.get_qvals(state)
            boltzmann_temp=1.
            distribution = torch.distributions.Categorical(logits=qvals/boltzmann_temp)
            action_idx = distribution.sample().item()
            action = np.atleast_1d(action_choices.take(action_idx))
        _next_state, reward, done, info = env.step(action)
        preprocessor.stream_state(_next_state)
        next_state = preprocessor.current_data()
        reward *= 10
        reward = max(min(reward, 1.), -1.)
        sarsd = SARSD(state, action, reward, next_state, done)
        rb.add(sarsd)
        # fig, ax = plot_sarsd(sarsd)
        # plt.show()
        if done:
            env.reset()
            preprocessor.initialize_history(env)
            state = preprocessor.current_data()
        else:
            state = next_state
        if len(rb) > config.min_rb_size:
            sarsd = rb.sample(config.batch_size)
            # import ipdb; ipdb.set_trace()
            metrics = agent.train_step(sarsd)
            train_metrics.append(metrics)
            if i % 100 == 0:
                print('step: ', i,
                        'eps: ', f"{eps:.2f}",
                        'loss: ', f"{train_metrics[-1]['loss']:.2e}",
                        'reward: ', f"{reward:.2f}",
                        f"Q_t: {train_metrics[-1]['Q_t'].mean():.4e}",
                        f"G_t: {train_metrics[-1]['G_t'].mean():.4e}",
                        'td_error: ', '{:.2e}'.format(metrics['td_error']))
        if i % target_update_freq ==0:
            agent.model_t.load_state_dict(agent.model_b.state_dict())

    post = list_2_dict(test(agent, env, preprocessor, nsteps=config.test_steps,
                           eps=0., random_starts=0, boltzmann=False, boltzmann_temp=1.,
                           verbose=True))
    # import ipdb; ipdb.set_trace()
    train_metrics = reduce_train_metrics(list_2_dict(train_metrics), columns=['Q_t', 'G_t'])

    qvals_match = [np.allclose(q, g) for q, g in zip(train_metrics['Q_t'], train_metrics['G_t'])]

    fig1, ax1 = plot_test_metrics(pre, include=('prices', 'equity', 'cash', 'positions',
                                                'margin', 'returns', 'actions',
                                                'qvals'))
    fig2, ax2 = plot_test_metrics(post, include=('prices', 'equity', 'cash', 'positions',
                                                 'margin', 'returns', 'actions',
                                                 'qvals'))
    fig3, ax3 = plot_train_metrics(train_metrics, include=('loss', 'td_error', 'G_t', 'Q_t',
                                                           'rewards'))
    plt.show()

if __name__=="__main__":
    # test_conv_forward_pass()
    # test_conv_random_batch_overfit()
    test_conv_synth_batch_overfit()
