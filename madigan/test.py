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



config = Config.from_exp(args.exp_id, args.basepath)
env = make_env(config)
agent = make_agent(config)
preprocessor = make_preprocessor(config, env.nFeats)

metrics = agent.test_episode(config.test_steps)
fig, ax = plot_test_metrics(metrics, assets=list(env.assets))
plt.show()
