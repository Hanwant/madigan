import numpy as np
import torch
from madigan.fleet import DQN
from madigan.environments import Synth
from madigan.utils.config import make_config
from madigan.fleet import make_agent
from madigan.environments import make_env
from madigan.run.train import Trainer
from madigan.run.test import test

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = make_config(
    experiment_id="Test",
    basepath="/media/hemu/Data/Markets/farm",
    nsteps=1_000_000,
    discrete_actions=True,
    discrete_action_atoms=11,
    min_tf=64,
    model_class="ConvModel",
    lr=1e-3,
    double_dqn=True,
    rb_size=100_000,
    train_freq=4,
    test_freq=32000,
    lot_unit_value=1000)
# env = make_env(config)
# agent = make_agent(config)
# trainer = Trainer(agent, env, config, print_progress=True)
trainer = Trainer.from_config(config, print_progress=True)
agent, env = trainer.agent, trainer.env
test_metrics_pre = test(agent, env, nsteps=1000, verbose=True)
trainer.train()
test_metrics_post = test(agent, env, nsteps=1000, verbose=True)

print('Done')
pre_eq = np.mean(test_metrics_pre['equity'])
post_eq = np.mean(test_metrics_post['equity'])
print(f'Equity over 1000 steps: pre/post training  {pre_eq}, {post_eq}')
