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



device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config",
                        help="config file for initializing experiment if manual testing",
                        default="/")
    arg = parser.parse_args()

    # import ipdb; ipdb.set_trace()
    config_path = Path(arg.config)
    config = load_config(config_path)


    trainer = Trainer.from_config(config, print_progress=True,
                                continue_exp=True, device=device)
    trainer.logger.setLevel(logging.INFO)
    agent, env = trainer.agent, trainer.env

