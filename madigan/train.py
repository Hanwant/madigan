import logging
import numpy as np
import torch
from madigan.fleet import DQN
from madigan.environments import Synth
from madigan.utils.config import make_config
from madigan.fleet import make_agent
from madigan.environments import make_env
from madigan.run.train import Trainer, plot_train_logs
from madigan.run.test import test, plot_episode, plot_test_logs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)
config = make_config(
    experiment_id="Test",
    basepath="/media/hemu/Data/Markets/farm",
    overwrite_exp=True,
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
    lot_unit_value=10_000)

trainer = Trainer.from_config(config, print_progress=True,
                              overwrite_logs=True)
agent, env = trainer.agent, trainer.env

test_episode_pre = test(agent, env, nsteps=1000, verbose=True)

# train_logs, test_logs = trainer.train(nsteps=60_000)

test_episode_post = test(agent, env, nsteps=1000, verbose=True)

print('Done')
pre_eq = np.mean(test_episode_pre['equity'])
post_eq = np.mean(test_episode_post['equity'])
print(f'Equity over 1000 steps: pre/post training  {pre_eq}, {post_eq}')
