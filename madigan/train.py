import sys
import logging
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

from madigan.fleet import make_agent
# from madigan.environments import  make_env
# from madigan.environments.cpp import Assets
from madigan.utils.config import load_config, make_config, save_config
from madigan.utils.plotting import plot_test_metrics, plot_train_metrics
# from madigan.utils.preprocessor import Preprocessor
from madigan.run.trainer import Trainer
# from madigan.run.test import test

parser = argparse.ArgumentParser()
parser.add_argument("config",
                    help="config file for initializing experiment if manual testing",
                    default="/")
parser.add_argument("--nsteps",
                    help="number of training_steps",
                    default=None)
# parser.add_argument("--continue_exp",
#                     help="Print metrics to stdout",
#                     action="store_true")
parser.add_argument("--verbose",
                    help="Print metrics to stdout",
                    action="store_true")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)

args = parser.parse_args()

config_path = Path(args.config)
config = load_config(config_path)
nsteps = int(args.nsteps)

trainer = Trainer.from_config(config, print_progress=True,
                              continue_exp=True, device=device)
agent, env = trainer.agent, trainer.env

pre = trainer.test()

train_logs, test_logs = trainer.train(nsteps=nsteps)

# import ipdb; ipdb.set_trace()

post = trainer.test()

print('Done')
print(f"Mean equity over 1000 steps: pre/post training  {np.mean(pre['equity'])}, {np.mean(post['equity'])}")
print(f"End equity after 1000 steps: pre/post training  {pre['equity'][-1]}, {post['equity'][-1]}")

# import ipdb; ipdb.set_trace()
fig1, ax1 = plot_test_metrics(pre, assets=config.assets)
fig2, ax2 = plot_test_metrics(post, assets=config.assets)
fig, ax = plot_train_metrics(trainer.load_logs()[0])


# fig1.show()
# fig2.show()
plt.show()
# sys.exit()





# assets=["OU1"],
# data_source_type="SineAdder",
# dat_source_config={
#     'freq':[2.2, 4.1, 1., 3.],
#     'mu':[.6, 0.3, 2., 4.2],
#     'amp':[.5, 0.2, 0.4, 1.2],
#     'phase':[0., 1., 4., 0.],
#     'dX':0.01,
#     "noise": 0.0},
# data_source_type="OU",
# generator_params=dict(
#     mean=[10.],
#     theta=[.15],
#     phi = [1.],
#     noise_var = [.1],
# ),
