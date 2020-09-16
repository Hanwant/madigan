import os
from pathlib import Path
from random import random
import logging
from typing import Union
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from ..utils import SARSD, State, ReplayBuffer, make_grid, save_to_hdf, load_from_hdf
from ..fleet import make_agent, DQN
from ..environments import make_env, Synth
from .test import test


def plot_train_logs(train_logs, filter_cols=('prices', )):
    metrics = list(filter(lambda x: x not in filter_cols,
                          train_logs.keys()))
    fig, axs = plt.subplots(*make_grid(len(metrics)))
    ax = axs.flatten()
    for i, name in enumerate(metrics):
        ax[i].plot(train_logs[name], label=name)
        ax[i].set_title(name)
        ax[i].legend()
    plt.show()

def list_2_dict(metrics: list):
    """
    aggregates a list of dicts (all with same keys) into a dict of lists

    the train_loop generator yield dictionaries of metrics at each iteration.
    this allows the loop to be interoperable in different scenarios
    The expense of getting a dict (instead of directly appending to list)
    is not too much but come back and PROFILE


    """
    if metrics is not None and len(metrics) > 0:
        if isinstance(metrics[0], dict):
            metrics = {k: [metric[k] for metric in metrics] for k in metrics[0].keys()}
            return metrics
    else:
        return {}

def reduce_train_metrics(metrics: Union[dict, pd.DataFrame], columns: list):
    """
    Takes dict (I.e from list_2_dict) or pandas df
    returns dict/df depending on input type
    """
    if metrics is not None and len(metrics):
        _metrics = type(metrics)() # Create a dict or pd df
        for col in metrics.keys():
            if col in columns:
                if isinstance(metrics[col][0], (np.ndarray, torch.Tensor)):
                    _metrics[col] = [m.mean().item() for m in metrics[col]]
                elif isinstance(metrics[col][0], list):
                    _metrics[col] = [np.mean(m).item() for m in metrics[col]]
                else:
                    _metrics[col] = metrics[col]
            else:
                _metrics[col] = metrics[col] # Copy might be needed for numpy arrays / torch tensors
    else:
        _metrics = metrics
    return _metrics

def reduce_test_metrics(test_metrics, cols=('returns', 'equity', 'cash', 'margin')):
    out = []
    if isinstance(test_metrics, dict):
        return list_2_dict(reduce_test_metrics([test_metrics], cols=cols))
    keys = test_metrics[0].keys()
    for m in test_metrics:
        _m = {}
        for k in keys:
            if k not in cols:
                _m[k] = m[k]
            else:
                if isinstance(m[k], (np.ndarray, torch.Tensor)):
                    _m[k] = m[k].mean().item()
                elif isinstance(m[k], list):
                    _m[k] = np.mean(m[k])
                try:
                    _m[k] = np.mean(m[k])
                except Exception as E:
                    import traceback
                    traceback.print_exc()
                    print("col passed to reduce_test_metrics did not contain ndarray/list/tensor")
                    print("np.mean tried anyway and failed")
        out.append(_m)
    return out

def get_train_loop(config):
    if config.agent_type == "DQN":
        return train_loop_dqn
    else:
        raise NotImplementedError("Only train loops for dqn are implemented")

class Trainer:
    """
    Orchestrates training and logging
    Instantiates from either instances of agent, env (+ config)
    or directly from config: trainer = Trainer.from_config(config)

    The central event loop is called via a generator.
    The Trainer.train() function runs through the generator
    Trainer.run_server() listens for jobs coming from a socket
    sending the results back
    """
    def __init__(self, agent, env, config, print_progress=True, overwrite_logs=False):
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logging.getLogger('trainer')
        self.print_progress = print_progress
        self.savepath = Path(config.basepath)/f'{config.experiment_id}/logs'
        self.logpath = self.savepath/'log.h5'
        if not self.savepath.is_dir():
            self.logger.info(f"Making New Experiment Directory {self.savepath}")
            self.savepath.mkdir(parents=True, exist_ok=True)
        self.train_loop = get_train_loop(config)
        iter(self)
        if overwrite_logs:
            if self.logpath.is_file():
                self.logger.warning("Overwriting previous log file")
                os.remove(self.logpath)
        self.test_metrics_cols = ('cash', 'equity', 'margin', 'returns')

    @classmethod
    def from_config(cls, config, **kwargs):
        agent = make_agent(config)
        env = make_env(config)
        return cls(agent, env, config, **kwargs)

    def save_logs(self, train_metrics, test_metrics, append=True):
        if len(train_metrics):
            train_metrics = list_2_dict(train_metrics)
            train_metrics = reduce_train_metrics(train_metrics, ['G_t', 'Q_t', 'rewards'])
            train_df = pd.DataFrame(train_metrics)
            # self.save_train_logs(train_df)
            save_to_hdf(self.logpath, 'train', train_df, append_if_exists=append)
        if len(test_metrics):
            test_metrics = reduce_test_metrics(test_metrics)
            test_metrics = dict(filter(lambda x: x[0] in self.test_metrics_cols, list_2_dict(test_metrics).items()))
            test_df = pd.DataFrame(test_metrics)
            save_to_hdf(self.logpath, 'test', test_df, append_if_exists=append)

    def load_logs(self):
        train_logs = load_from_hdf(self.logpath, 'train')
        test_logs = load_from_hdf(self.logpath, 'test')
        return train_logs, test_logs

    def __iter__(self):
        self.train_loop = iter(self.train_loop(self.agent, self.env, self.config, print_progress=self.print_progress))
        return self

    def __next__(self):
        """
        Returns a 'train_metric' which is either a dict or None:
        """
        return next(self.train_loop)

    def test(self, nsteps=None):
        nsteps = nsteps or self.config.test_steps
        episode_metrics = test(self.agent, self.env, nsteps=nsteps)
        return episode_metrics

    def train(self, nsteps=None):
        """
        Wraps the train_loop generator
        Performs funcitons which are not essential to model optimizaiton
        This includes:
        - accumulating logs of training metrics
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
        log_freq=self.config.log_freq
        test_freq = self.config.test_freq
        model_save_freq = self.config.model_save_freq
        nsteps = nsteps or self.config.nsteps
        train_metrics = []
        test_metrics = []
        i = 0
        try:
            self.logger.info("Starting Training")
            test_metric = test(self.agent, self.env, self.config.test_steps)
            self.agent.save_state()
            while i < nsteps:
                train_metric = next(self.train_loop)

                if i % test_freq == 0:
                    self.logger.info("Testing Model")
                    test_metric = test(self.agent, self.env, self.config.test_steps)

                if i % model_save_freq == 0:
                    self.logger.info("Saving Model")
                    self.agent.save_state()

                if train_metric is not None:
                    train_metrics.append(train_metric)

                if test_metric is not None:
                    test_metrics.append(test_metric)
                    test_metric=None

                if i % log_freq == 0:
                    self.logger.info("Logging Metrics")
                    self.save_logs(train_metrics, test_metrics)
                    train_metrics, test_metrics = [], []
                i+=1

        except StopIteration:
            print('Done training')

        except KeyboardInterrupt:
            import ipdb; ipdb.set_trace()

        except Exception as E:
            import traceback
            traceback.print_exc()
            import ipdb; ipdb.set_trace()

        finally:
            test_metrics.append(test(self.agent, self.env, self.config.test_steps))
            self.agent.training_steps += train_metric['training_steps']
            self.agent.total_steps += train_metric['total_steps']
            self.agent.save_state()
            self.save_logs(train_metrics, test_metrics)
            train_metrics, test_metrics = self.load_logs()
            return train_metrics, test_metrics

def train(agent, env, config, nsteps=None):
    """
    Skeleton for wrapping a train loop
    """
    train_loop = get_train_loop(config)
    loop = iter(train_loop(agent, env, config))
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
                test_metric = test(agent, env, config.test_steps)

            if i % model_save_freq == 0:
                agent.save_state()

            if train_metric is not None:
                train_metrics.append(train_metric)

            if test_metric is not None:
                test_metrics.append(test_metric)
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

def train_loop_dqn(agent, env, config, print_progress=True):
    """
    Encapsulates logic which affects model optimization
    Is wrapped by handlers which take care of logging etc
    """

    rb = ReplayBuffer(config.rb_size)

    eps = config.expl_eps
    eps_min = config.expl_eps_min
    eps_decay = config.expl_eps_decay

    state = env.reset()
    I = agent.total_steps
    nsteps = I + config.nsteps
    eps *= (1-eps_decay)**I

    if print_progress:
        iterator = tqdm(range(I, nsteps))
    else:
        iterator = range(I, nsteps)

    min_rb_size = config.min_rb_size
    tgt_update_freq = config.target_update_freq
    train_freq = config.train_freq

    train_metric = None
    local_steps = 0
    steps_since_train = 0
    training_steps = 0
    steps_since_target_update = 0
    for i in iterator:

        if random() < eps:
            action = env.action_space.sample()
        else:
            action = agent(state)
        eps *= (1-eps_decay)

        next_state, reward, done, info = env.step(action)
        rb.add(SARSD(state, action, reward, next_state, done))
        state = next_state

        if done:
            state = env.reset()

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
    agent.training_steps += training_steps
    agent.total_steps += i



