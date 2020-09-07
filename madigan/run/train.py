from pathlib import Path
from random import random
import logging
from typing import Union
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from ..utils import SARSD, State, ReplayBuffer
from ..fleet import make_agent, DQN
from ..environments import make_env, Synth
from .test import test

logger = logging.getLogger('trainer')
logger.setLevel(logging.DEBUG)

def pack_metrics(metrics: list):
    """
    This function aggregates a list of (homogenous-keys) dicts into a dict of lists

    the train_loop generator yield dictionaries of metrics at each iteration.
    this allows the loop to be interoperable in different scenarios
    The expense of getting a dict (instead of directly appending to list)
    is not too much but come back and PROFILE


    """
    if len(metrics) > 0:
        if isinstance(metrics[0], dict):
            metrics = {k: [metric[k] for metric in metrics] for k in metrics[0].keys()}
    return metrics

def reduce_metrics(metrics: Union[dict, pd.DataFrame], columns: list):
    """
    Takes dict (I.e from pack_metrics) or pandas df
    returns dict/df depending on input type
    """
    if len(metrics):
        _metrics = type(metrics)() # Create a dict or pd df
        for col in metrics.keys():
            if col in columns:
                if isinstance(metrics[col][0], (np.ndarray, torch.Tensor)):
                    _metrics[col] = [m.mean().item() for m in metrics[col]]
                else:
                    _metrics[col] = metrics[col]
            else:
                _metrics[col] = metrics[col] # Copy might be needed for numpy arrays / torch tensors
    else:
        _metrics = metrics
    return _metrics

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
    def __init__(self, agent, env, config, print_progress=True):
        self.agent = agent
        self.env = env
        self.config = config
        self.print_progress = print_progress
        self.savepath = Path(config.basepath)/f'{config.experiment_id}/logs'
        if not self.savepath.is_dir():
            self.savepath.mkdir(parents=True, exist_ok=True)
        self = iter(self)

    @classmethod
    def from_config(cls, config, print_progress=True):
        agent = make_agent(config)
        env = make_env(config)
        return cls(agent, env, config, print_progress=print_progress)


    def save_logs(self, train_metrics, test_metrics):
        train_metrics = pack_metrics(train_metrics)
        test_metrics = pack_metrics(test_metrics)
        train_metrics = reduce_metrics(train_metrics, ['G_t', 'Q_t', 'rewards'])
        # test_metrics = reduce_metrics(test_metrics, [''])
        train_df = pd.DataFrame(train_metrics)
        # test_df = pd.DataFrame(test_metrics)
        with pd.HDFStore(self.savepath/'logs.h5', mode='a') as f:
            if len(train_df):
                if 'train' not in f.keys():
                    f.put('train', train_df, format='t')
                else:
                    train_df.to_hdf(f, key='train', mode='a', format='t')
            # if len(test_df):
            #     if 'test' not in f.keys():
            #         f.put('test', test_df, format='t')
            #     else:
            #         test_df.to_hdf(f, key='test', mode='a', format='t')

    def load_logs(self):
        train_metrics, test_metrics = None, None
        with pd.HDFStore(self.savepath/'logs.h5', 'r') as f:
            if '/train' in f.keys():
                train_metrics = pd.read_hdf(f, key='train', mode='r')
            # if '/test' in f.keys():
            #     test_metrics = pd.read_hdf(f, key='test', mode='r')
        return train_metrics, test_metrics

    def __iter__(self):
        self.train_loop = iter(train_loop(self.agent, self.env, self.config, print_progress=self.print_progress))
        return self

    def __next__(self):
        """
        Returns a tuple:
        (train_metric, test_metric)
        check for None
        """
        return next(self.train_loop)

    def train(self, print_progress=True):
        train_metrics = []
        test_metrics = []
        i=0
        log_freq=self.config.log_freq
        try:
            while True:
                train_metric, test_metric = next(self.train_loop)
                if train_metric is not None:
                    train_metrics.append(train_metric)
                if test_metric is not None:
                    test_metrics.append(test_metric)
                if i % log_freq == 0:
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
            train_metrics, test_metrics = self.load_logs()
            return train_metrics, test_metrics


def train_loop(agent, env, config, print_progress=True):

    rb = ReplayBuffer(config.rb_size)

    eps = config.expl_eps
    eps_min = config.expl_eps_min
    eps_decay = config.expl_eps_decay

    state = env.reset()
    I = agent.total_steps
    nsteps = I + config.nsteps
    eps *= (1-eps_decay)**I
    local_steps = 0
    steps_since_target_update = 0
    steps_since_train = 0

    if print_progress:
        iterator = tqdm(range(I, nsteps))
    else:
        iterator = range(I, nsteps)

    train_metric = None
    test_metric = None
    logger.debug('Initialized loop variables, starting loop')
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
            # print('Done', end='\r', flush=True)
            # print(info['Event'], end='\r', flush=True)
            state = env.reset()

        if len(rb) >= config.min_rb_size:
            if steps_since_train > config.train_freq:
                data = rb.sample(config.batch_size)
                train_metric = agent.train_step(data)
                train_metric['rewards'] = data.reward
                steps_since_train = 0
            if steps_since_target_update > config.test_freq:
                test_metric = test(agent, env, config.test_steps)
                agent.model_t.load_state_dict(agent.model_b.state_dict())
                steps_since_target_update = 0
            steps_since_train += 1
            steps_since_target_update += 1
        local_steps += 1
        yield train_metric, test_metric


def train(agent, env, config):
    loop = iter(train_loop(agent, env, config))
    train_metrics = []
    test_metrics = []
    try:
        while True:
            train_metric, test_metric = next(loop)
            if train_metric is not None:
                train_metrics.append(train_metric)
            if test_metric is not None:
                test_metrics.append(test_metric)

    except StopIteration:
        print('Done training')

    except KeyboardInterrupt:
        import ipdb; ipdb.set_trace()

    except Exception as E:
        import traceback
        traceback.print_exc()

    return train_metrics, test_metrics

