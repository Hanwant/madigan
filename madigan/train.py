import sys
import logging
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

from madigan.modelling import make_agent
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
    parser.add_argument("--nsteps",
                        help="number of training_steps",
                        default=None)
    parser.add_argument("--wandb",
                        help="log results to wandb",
                        action="store_true")
    parser.add_argument("--verbose",
                        help="Print metrics to stdout",
                        action="store_true")
    arg = parser.parse_args()

    # import ipdb; ipdb.set_trace()
    config_path = Path(arg.config)
    config = load_config(config_path)

    if arg.wandb:
        import wandb
        wandb.login()
        wandb.init(project="madigan", name=config.experiment_id,
                id=config.experiment_id, config=config,
                tags=[config.data_source_type, config.agent_type],
                dir=config.basepath, allow_val_change=True, resume=True)

    nsteps = int(arg.nsteps)

    trainer = Trainer.from_config(config, print_progress=True,
                                continue_exp=True, device=device)
    trainer.logger.setLevel(logging.INFO)
    agent, env = trainer.agent, trainer.env

    # pre = trainer.test()

    train_logs, test_logs = trainer.train(nsteps=nsteps)


    # print('Done')
    # print(f"Mean equity over 1000 steps: pre/post training  {np.mean(pre['equity'])}, {np.mean(test_logs['equity'])}")
    # print(f"End equity after 1000 steps: pre/post training  {pre['equity'].iloc[-1]}, {test_logs['equity'].iloc[-1]}")


    # fig1, ax1 = plot_test_metrics(pre, assets=config.assets)
    # fig2, ax2 = plot_test_metrics(test_logs, assets=config.assets)
    # fig, ax = plot_train_metrics(train_logs)

    # if arg.wandb:
    #     wandb.log({'pre training test_episode': fig1})
    #     wandb.log({'post training test episode': fig1})
    #     wandb.log({'training': fig})


    # # fig1.show()
    # # fig2.show()
    # plt.show()
    # # sys.exit()





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
