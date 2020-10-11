import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch

from madigan.fleet import DQN, make_agent
from madigan.environments import Synth, make_env
from madigan.environments.cpp import Assets
from madigan.utils.config import make_config, save_config, load_config
from madigan.utils.plotting import plot_test_metrics, plot_train_metrics
from madigan.utils.preprocessor import Preprocessor
from madigan.run.train import Trainer
from madigan.run.test import test

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)

config = make_config(
    experiment_id="Test",
    basepath="/media/hemu/Data/Markets/farm",
    overwrite_exp=True,
    nsteps=10_000_000,
    assets=["sine1"],
    required_margin=1.,

    preprocessor_type="WindowedStacker",
    window_length=64,

    agent_type="DQN",
    discrete_actions=True,
    discrete_action_atoms=3,
    lot_unit_value=1000,
    double_dqn=True,
    expl_eps=1.,
    expl_eps_min=0.1,
    expl_eps_decay=1e-6,

    model_class="ConvModel",
    lr=1e-3,
    d_model=64, # dimensionality of model
    n_layers=4, # number of layer units
    n_feats=1, # 1 corresponds to an input of just price

    batch_size=32,
    rb_size=100_000,
    train_freq=4,
    test_freq=32000,
    generator_params={
        'freq':[1.],
        'mu':[1.1],
        'amp':[1.],
        'phase':[0.],
        'dX':0.01}
)
config.save()

trainer = Trainer.from_config(config, print_progress=True,
                              overwrite_logs=config.overwrite_exp)
agent, env = trainer.agent, trainer.env
preprocessor = Preprocessor.from_config(config)

pre = test(agent, env, preprocessor, eps=0.1, nsteps=1000, verbose=True)

train_logs, test_logs = trainer.train(nsteps=400_000)

post = test(agent, env, preprocessor, eps=0.1, nsteps=1000, verbose=True)

print('Done')
print(f"Mean equity over 1000 steps: pre/post training  {np.mean(pre['equity'])}, {np.mean(post['equity'])}")
print(f"End equity after 1000 steps: pre/post training  {pre['equity'][-1]}, {post['equity'][-1]}")

# import ipdb; ipdb.set_trace()
fig1, ax1 = plot_test_metrics(pre, assets=config.assets)
fig2, ax2 = plot_test_metrics(post, assets=config.assets)
# fig, ax = plot_train_metrics(trainer.load_logs()[0])


# fig1.show()
# fig2.show()
plt.show()
# sys.exit()
