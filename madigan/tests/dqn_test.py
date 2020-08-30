import pytest
import numpy as np
from madigan.environments import Synth
from madigan.fleet.dqn import DQN
from madigan.utils import make_config, State, SARSD

@pytest.mark.skip("mlp not implemented yet")
def test_forward_pass_mlp():
    config = make_config(model_class="MLPModel", nassets=4, discrete_actions=True,
                         discrete_action_atoms=11)
    in_shape = config['agent_config']['model_config']['in_shape']
    out_shape = config['agent_config']['model_config']['out_shape']
    dqn = DQN(config, name="test")
    state = np.empty(in_shape)[None, ...]
    action = dqn(state, target=True)
    qvals = dqn(state, raw_qvals=True)
    assert qvals.shape[1:] == out_shape

def test_conv_forward_pass():
    config = make_config(discrete_actions=True, discrete_action_atoms=11, min_tf=12,
                         model_class="ConvModel")
    dqn = DQN(config, name="test")
    n_feats = config['agent_config']['model_config']['n_feats']
    price_shape = (1, config['n_assets'], config['min_tf'])
    port_shape = (1, config['n_assets'])
    price = np.random.randn(*price_shape)
    port = np.random.randn(*port_shape)
    state = State(price=price, port=port)
    action = dqn(state, target=True)


def test_conv_batch_overfit():
    config = make_config(discrete_actions=True, discrete_action_atoms=11, min_tf=12,
                         model_class="ConvModel")
    dqn = DQN(config, name="test")
    env = Synth(**config)
    state = env.reset()
    action = env.action_space.sample()
    next_state, reward, done, info = env.step(action)
    sarsd = SARSD(state, action, reward, next_state, done)
    for i in range(100):
        metrics = dqn.train_step(sarsd)
        print(metrics['loss'])


if __name__=="__main__":
    test_conv_forward_pass()
    test_conv_batch_overfit()
