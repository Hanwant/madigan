import sys
import logging
from functools import partial
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

from madigan.fleet import DQN, make_agent
from madigan.environments import Synth, make_env
from madigan.environments.cpp import Assets
from madigan.utils.config import make_config, Config
from madigan.utils.plotting import plot_test_metrics, plot_train_metrics
from madigan.utils.preprocessor import make_preprocessor
from madigan.run.test import test, test_manual

logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("exp_id",
                    help="Either id for existing experiment or 'manual' for manual testing")
parser.add_argument("--basepath",
                    help="path where all experiments are stored",
                    default="/media/hemu/Data/Markets/farm")
parser.add_argument("--config",
                    help="config file for initializing experiment if manual testing",
                    default="/media/hemu/Data/Markets/farm")
parser.add_argument("--verbose",
                    help="Print metrics to stdout",
                    action="store_true")
args = parser.parse_args()

if args.exp_id == "manual":
    config = make_config(
        experiment_id="Test",
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
    env = make_env(config)
    preprocessor = make_preprocessor(config)
    tester = partial(test_manual, env, preprocessor, verbose=args.verbose)

else:
    config = Config.from_exp(args.exp_id, args.basepath)
    env = make_env(config)
    agent = make_agent(config)
    preprocessor = make_preprocessor(config)
    tester = partial(test, agent, env, preprocessor, nsteps=config.test_steps,
                     eps=0., random_starts=0, verbose=args.verbose)



metrics = tester()
fig, ax = plot_test_metrics(metrics, assets=config.assets)
plt.show()
