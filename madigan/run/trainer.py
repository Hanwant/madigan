import os
from pathlib import Path
from random import random
import logging
from typing import Union
import yaml

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ..utils import SARSD, State, ReplayBuffer, save_to_hdf, load_from_hdf
from ..utils.metrics import list_2_dict, reduce_train_metrics, test_summary
from ..utils.config import save_config
from ..utils.plotting import make_grid
from ..utils.preprocessor import make_preprocessor
from ..fleet import make_agent, DQN
from ..environments import make_env, get_env_info
from ..environments.cpp import RiskInfo
from .test import test


class Trainer:
    """
    Orchestrates training and logging
    Instantiates from either instances of agent (+ config)
    or directly from config: trainer = Trainer.from_config(config)

    The central event loop is called via a generator.
    The Trainer.train() function runs through the generator
    Trainer.run_server() listens for jobs coming from a socket
    sending the results back
    """
    def __init__(self, agent, config, print_progress=True, continue_exp=True,
                 device=None):
        device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = agent
        self.agent.to(device)
        self.env = agent.env
        self.preprocessor = agent.preprocessor
        self.config = config
        self.train_steps = config.train_steps
        self.logger = logging.getLogger(__name__)
        self.log_freq = config.log_freq
        self.test_freq = config.test_freq
        self.print_progress = print_progress
        self.logdir = Path(config.experiment_path) / 'logs'
        self.logfile = self.logdir/'log.h5'
        if not self.logdir.is_dir():
            self.logger.info("Making New Experiment Directory %s", self.logdir)
            self.logdir.mkdir(parents=True, exist_ok=True)
        # iter(self)
        if not continue_exp:
            if self.logfile.is_file():
                self.logger.warning("Overwriting previous log file")
                os.remove(self.logfile)
        # self.test_metrics_cols = ('cash', 'equity', 'margin', 'returns')
        self.test_metrics_cols = None # equivalent to saving all mertrics
        self.config.save()

    @classmethod
    def from_config(cls, config, device=None, **kwargs):
        agent = make_agent(config)
        return cls(agent, config, device=device, **kwargs)

    def save_logs(self, train_metrics, test_metrics, append=True):
        if len(train_metrics):
            train_metrics = list_2_dict(train_metrics)
            train_metrics = reduce_train_metrics(train_metrics, ['Gt', 'Qt', 'rewards'])
            train_df = pd.DataFrame(train_metrics)
            save_to_hdf(self.logdir/'train.hdf5', 'train', train_df,
                        append_if_exists=append)
        if len(test_metrics):
            # if self.test_metrics_cols is not None:
            #     test_metrics = dict(filter(lambda x: x[0] in self.test_metrics_cols,
            #                                list_2_dict(test_metrics).items()))
            self.logger.info(f'logging {len(test_metrics)} test runs')
            for (env_step, test_run) in test_metrics:
                test_df = pd.DataFrame(test_run)
                test_filename = self.logdir/(f'test_env_steps_{env_step}'+\
                    f'_episode_steps_{len(test_df)}.hdf5')
                test_df.to_hdf(test_filename, 'full_run', append=False)
                summary = test_summary(test_df)
                summary['env_steps'] = [self.agent.env_steps]
                summary['training_steps'] = [self.agent.training_steps]
                # save_to_hdf(self.logdir/'test.hdf5', 'run_history',
                #             summary, append_if_exists=True)
                summary.to_hdf(self.logdir/'test.hdf5', 'run_history',
                               append=True)

    def load_logs(self):
        train_logs = pd.read_hdf(self.logdir/'train.hdf5', 'train')
        test_logs = self.load_latest_test_run()
        return train_logs, test_logs

    def load_latest_test_run(self):
        files = list(filter(lambda x: 'test' in x.stem, self.logdir.iterdir()))
        files = sorted([(int(f.stem.split('_')[3]), f) for f in files])
        return pd.read_hdf(files[-1][1], key='full_run')


    def test(self, test_steps=None, reset=True):
        test_steps = test_steps or self.config.test_steps
        # episode_metrics = test(self.agent, self.env, self.preprocessor, nsteps=nsteps)
        test_metrics = self.agent.test_episode(test_steps=test_steps, reset=reset)
        return pd.DataFrame(test_metrics)

    def train(self, nsteps=None):
        """
        Wraps train loop of agent (agent.step(n))
        Performs funcitons which are not essential to model optimizaiton
        This includes:
        - accumulating and saving training metrics
        - periodically rolling out test episodes
        - accumulating logs of test metrics
        - saving logs to hdf5 files
        - saving models
        - Exception handling

        Params
        nsteps: int number of steps to run training for. This is limited by the
                config.nsteps parameter. If None, config.nsteps is used. default = None

        returns tuple of pandas df (train_logs, test_logs)

        """
        flush_freq = self.log_freq
        test_freq = self.test_freq
        model_save_freq = self.config.model_save_freq
        nsteps = nsteps or self.train_steps
        train_metrics = []
        test_metrics = []
        i = self.agent.training_steps
        steps_since_test = 0
        steps_since_save = 0
        steps_since_flush = 0

        # fill replay buffer up to replay_min_size before training
        self.logger.info('filling replay buffer to min size before training')
        burn_in_train_loop = iter(self.agent.step(self.agent.replay_min_size))
        train_metrics.extend(next(burn_in_train_loop))
        test_metrics.append((i, self.agent.test_episode()))

        self.logger.info("Starting Training")
        train_loop = self.agent.step(nsteps)

        progress_bar = tqdm(total=i+nsteps, colour='#9fc693')
        progress_bar.update(i)
        try:
            while True:
                train_metric = next(train_loop)
                train_metrics.extend(train_metric)
                training_steps_taken = self.agent.training_steps - i
                progress_bar.update(training_steps_taken)
                i = self.agent.training_steps

                if steps_since_test > test_freq:
                    self.logger.info("Testing Model")
                    test_metrics.append((i, self.agent.test_episode()))
                    steps_since_test = 0

                if steps_since_save > model_save_freq:
                    self.logger.info("Saving Agent State")
                    self.agent.save_state()
                    steps_since_save = 0

                if steps_since_flush > flush_freq:
                    self.logger.info("Saving Log Metrics")
                    self.save_logs(train_metrics, test_metrics)
                    train_metrics, test_metrics = [], []
                    steps_since_flush = 0

                steps_since_test += training_steps_taken
                steps_since_save += training_steps_taken
                steps_since_flush += training_steps_taken

        except StopIteration:
            print('Done training')
            pass

        except KeyboardInterrupt:
            import ipdb; ipdb.set_trace()
            pass

        except Exception as E:
            import traceback
            traceback.print_exc()
            import ipdb; ipdb.set_trace()

        finally:
            test_metrics.append((i, self.agent.test_episode()))
            self.agent.save_state()
            self.save_logs(train_metrics, test_metrics)
            train_metrics, test_metrics = self.load_logs()
            progress_bar.close()
            return train_metrics, test_metrics



def train(agent, env, preprocessor, config, nsteps=None):
    """
    Skeleton for wrapping a train loop
    """
    train_loop = get_train_loop(config)
    loop = iter(train_loop(agent, env, preprocessor, config))
    train_metrics = []
    test_metrics = []
    test_freq = config.test_freq
    model_save_freq = config.model_save_freq
    nsteps = nsteps or config.nsteps
    try:
        i = 0
        while i < nsteps:
            train_metric = next(loop)

            if i % test_freq == 0:
                test_metric = test(agent, env, preprocessor,
                                   config.test_steps)

            if i % model_save_freq == 0:
                agent.save_state()

            if train_metric is not None:
                train_metrics.extend(train_metric)

            if test_metric is not None:
                test_metrics.extend(test_metric)
                test_metric=None
            i += 1

    except StopIteration:
        print('Done training')

    except KeyboardInterrupt:
        import ipdb; ipdb.set_trace()

    except Exception as E:
        import traceback
        traceback.print_exc()

    return train_metrics, test_metrics

def format_env_info(info_dict):
    """ for printing """
    # return yaml.dump(info_dict, default_flow_style=None, sort_keys=False)
    formatted=""
    for k, v in info_dict.items():
        formatted += k + ": " + repr(v) + "\n"
    return formatted


def train_loop_dqn(agent, env, preprocessor, config, print_progress=True):
    """
    Encapsulates logic which affects model optimization
    Is wrapped by handlers which take care of logging etc
    """

    rb = ReplayBuffer(config.rb_size)

    eps = config.expl_eps
    eps_min = config.expl_eps_min
    eps_decay = config.expl_eps_decay

    I = agent.total_steps
    nsteps = I + config.nsteps
    eps *= (1-eps_decay)**I

    if print_progress:
        iterator = tqdm(range(I, nsteps))
    else:
        iterator = range(I, nsteps)

    min_rb_size = config.min_rb_size
    min_tf = config.min_tf
    tgt_update_freq = config.target_update_freq
    train_freq = config.train_freq

    train_metric = None
    local_steps = 0
    steps_since_train = 0
    training_steps = 0
    steps_since_target_update = 0

    _state = env.reset()
    preprocessor.stream_state(_state)
    preprocessor.initialize_history(env) # runs empty actions until enough data is accumulated
    state = preprocessor.current_data()

    for i in iterator:

        if random() < eps:
            action = agent.action_space.sample()
        else:
            action = agent(state)
        eps *= (1-eps_decay)

        prev_metrics = get_env_info(env)
        _next_state, _reward, done, info = env.step(action)
        reward = (env.equity-prev_metrics['equity'])/prev_metrics['equity']
        if info.brokerResponse.marginCall: # manual check required as .riskInfo contains
            done = True                    # info for the transaction which may closing a position
        if abs(reward) > 1.:
            print("ABS REWARD > 1")
            print("reward: ", reward)
            print("done: ", done,)
            print("margin_call: ", info.brokerResponse.marginCall)
            print("action: ", info.brokerResponse.transactionUnits)
            print("riskInfo: ", info.brokerResponse.riskInfo, "\n")
            print('prev_metrics: ', format_env_info(prev_metrics))
            print('current_metrics: ', format_env_info(get_env_info(env)))
        reward = max(min(reward, 1.), -1.)
        # if done:
        #     reward = -1.
        preprocessor.stream_state(_next_state)
        next_state = preprocessor.current_data()
        rb.add(SARSD(state, action, reward, next_state, done))
        state = next_state

        if done:
            _state = env.reset()
            preprocessor.stream_state(_state)
            preprocessor.initialize_history(env)
            state = preprocessor.current_data()

        if len(rb) >= min_rb_size:
            if steps_since_train >= train_freq:
                data = rb.sample(config.batch_size)
                train_metric = agent.train_step(data)
                train_metric['rewards'] = data.reward
                training_steps += 1
                train_metric['training_steps'] = training_steps
                train_metric['total_steps'] = i
                steps_since_train = 0
            if steps_since_target_update >= tgt_update_freq:
                agent.target_update()
            steps_since_train += 1
            steps_since_target_update += 1
        local_steps += 1
        yield train_metric
        train_metric = None
    # agent.training_steps += training_steps
    # agent.total_steps += i



