import pytest
import numpy as np
import torch
from madigan.environments import Synth
from madigan.fleet.dqn import DQN
from madigan.utils import make_config, State, SARSD, batchify_sarsd

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

@pytest.mark.skip("util")
def make_data(config, sample_size=1):
    price_shape = (sample_size, config['min_tf'], config['n_assets'])
    port_shape = (sample_size, config['n_assets'])
    price = np.random.randn(*price_shape)
    port = np.random.randn(*port_shape)
    state = State(price, port)
    action = np.random.randint(0, config.agent_config.action_atoms, sample_size)
    reward = np.random.randn(sample_size)
    price = np.random.randn(*price_shape)
    port = np.random.randn(*port_shape)
    next_state = State(price, port)
    done = np.random.binomial(1, 0.5, sample_size)
    return SARSD(state, action, reward, next_state, done)



def test_conv_forward_pass():
    config = make_config(discrete_actions=True, discrete_action_atoms=11, min_tf=12,
                         model_class="ConvModel")
    dqn = DQN(config, name="test")
    n_feats = config['agent_config']['model_config']['n_feats']

    # price_shape is (bs, tf, feats) - tf and feats are switched inside model for convolution
    price_shape = (1, config['min_tf'], config['n_assets'])
    port_shape = (1, config['n_assets'])

    price = np.random.randn(*price_shape)
    port = np.random.randn(*port_shape)
    state = State(price=price, port=port)
    action = dqn(state, target=True)
    assert tuple(action.shape) == (1, config.n_assets)


def train_step(self, sarsd, device):
    import torch
    import torch.nn.functional as F
    self.opt.zero_grad()

    with torch.no_grad():
        if self.double_dqn:
            pass
        else:
            next_states = State(torch.tensor(sarsd.next_state.price, dtype=torch.float32).to(device),
                                torch.tensor(sarsd.next_state.port, dtype=torch.float32).to(device))
            qvals_next = self.model_t(next_states).max(-1)[0]
        rewards = torch.tensor(sarsd.reward, dtype=torch.float32, device=device).unsqueeze(-1)
        done_mask = ~torch.tensor(sarsd.done, dtype=torch.bool, device=device).unsqueeze(-1)
        G_t = rewards + (self.discount * done_mask * qvals_next)

    actions = torch.tensor(sarsd.action, dtype=torch.long, device=device).unsqueeze(-1)
    actions_mask = F.one_hot(actions, self.action_atoms).to(device)
    states = State(torch.tensor(sarsd.state.price, dtype=torch.float32).to(device),
                        torch.tensor(sarsd.state.port, dtype=torch.float32).to(device))
    qvals = self.model_b(states)
    Q_t = (qvals * actions_mask).sum(-1)
    loss = ((G_t - Q_t)**2).mean()
    loss.backward()
    self.opt.step()
    return {'loss': loss.detach().item(), 'td_error': 0., 'G_t': G_t, 'Q_t': Q_t}




@pytest.mark.skip()
def test_conv_batch_overfit():
    config = make_config(discrete_actions=True, discrete_action_atoms=11, min_tf=64,
                         model_class="ConvModel", lr=1e-3, double_dqn=False)
    agent = DQN(config, name="test")
    device = 'cuda'
    # env = Synth(**config)
    # state = env.reset()
    # action = env.action_space.sample()
    # next_state, reward, done, info = env.step(action)
    # sarsd = batchify_sarsd(SARSD(state, action, reward, next_state, done))
    sarsd = make_data(config, sample_size=32)
    qvals = []
    for i in range(2000):
        # metrics = train_step(agent, sarsd, device)
        metrics = agent.train_step(sarsd)
        qvals.append([metrics['G_t'], metrics['Q_t']])
        if i % 100 == 0:
            print('step: ', i, 'loss: ', f"{metrics['loss']:.2e}", 'td_error: ', '{:.2e}'.format(metrics['td_error']))
    qvals_match = [torch.allclose(q[0], q[1]) for q in qvals]
    assert sum(qvals_match) > 1




if __name__=="__main__":
    test_conv_forward_pass()
    test_conv_batch_overfit()
