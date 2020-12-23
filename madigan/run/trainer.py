import os
import shutil
from pathlib import Path
from random import random
import logging
from typing import Union
from queue import Queue
import yaml

from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import zmq

from ..utils.data import SARSD, State
from ..utils.replay_buffer import ReplayBuffer
from ..utils.logging import save_to_hdf, load_from_hdf
from ..utils.metrics import list_2_dict, reduce_train_metrics, test_summary
from ..utils.config import save_config, Config
from ..utils.plotting import make_grid
from ..utils.preprocessor import make_preprocessor
from ..modelling import make_agent
from ..environments import make_env, get_env_info
from ..environments.cpp import RiskInfo
# from .test import test


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
        self.test_steps = config.test_steps
        self.logger = logging.getLogger(__name__)
        self.log_freq = config.log_freq
        self.test_freq = config.test_freq
        self.print_progress = print_progress
        self.logdir = Path(config.basepath)/config.experiment_id/'logs'
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
        self.server_host = 9000
        self.server_port = 9000
        # self.init_server()
        self.terminate_early = False

    def init_server(self, port: int = None):
        port = port or self.server_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.setsockopt(zmq.LINGER, 0)
        # self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f'tcp://{self.server_host}:{port}')
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN | zmq.POLLOUT)
        self.recvque = Queue()

    @classmethod
    def from_config(cls, config, device=None, **kwargs):
        agent = make_agent(config)
        return cls(agent, config, device=device, **kwargs)

    def save_logs(self,
                  train_metrics: Union[dict, pd.DataFrame],
                  test_metrics: Union[dict, pd.DataFrame],
                  append: bool = True):
        if len(train_metrics):
            train_metrics = list_2_dict(train_metrics)
            train_metrics = reduce_train_metrics(train_metrics, ['Gt', 'Qt', 'rewards',
                                                                 'entropy'])
            train_df = pd.DataFrame(train_metrics)
            # save_to_hdf(self.logdir/'train.hdf5', 'train', train_df,
            #             append_if_exists=append)
            train_df.to_hdf(self.logdir/'train.hdf5', 'train', append=append)
        if len(test_metrics):
            # if self.test_metrics_cols is not None:
            #     test_metrics = dict(filter(lambda x: x[0] in self.test_metrics_cols,
            #                                list_2_dict(test_metrics).items()))
            self.logger.info(f'logging {len(test_metrics)} test runs')
            for (env_step, test_run) in test_metrics:
                test_df = pd.DataFrame(test_run)
                test_filename = self.logdir/(
                    f'test_env_steps_{env_step}' +
                    f'_episode_steps_{len(test_df)}.hdf5')
                test_df.to_hdf(test_filename, 'full_run', append=False)
                with h5py.File(test_filename, 'a') as f:
                    f.attrs['asset_names'] = [asset.code for asset in
                                              self.agent.env.assets]
                summary = test_summary(test_df)
                summary['env_steps'] = [self.agent.env_steps]
                summary['training_steps'] = [self.agent.training_steps]
                # save_to_hdf(self.logdir/'test.hdf5', 'run_history',
                #             summary, append_if_exists=True)
                summary.to_hdf(self.logdir/'test.hdf5', 'run_history',
                               append=True)

    def load_logs(self):
        train_logs = pd.read_hdf(self.logdir/'train.hdf5', 'train')
        # test_logs = self.load_latest_test_run()
        test_logs = pd.read_hdf(self.logdir/'test.hdf5', 'run_history')
        return train_logs, test_logs

    def load_latest_test_run(self):
        files = list(filter(lambda x: 'test_' in x.stem, self.logdir.iterdir()))
        files = sorted([(int(f.stem.split('_')[3]), f) for f in files])
        return pd.read_hdf(files[-1][1], key='full_run')

    def test(self, test_steps=None, reset=True):
        test_steps = test_steps or self.test_steps
        test_metrics = self.agent.test_episode(test_steps=test_steps,
                                               reset=reset)
        return pd.DataFrame(test_metrics)

    def revert_to_checkpoint(self, checkpoint_id: str):
        raise NotImplementedError("reverting to checkpoint for main not impl")

    def branch_from_checkpoint(self, checkpoint: Union[str, int]):
        """
        Creates a child branch using a previous checkpoint as a branching point
        checkpoint: may be either an integer (# of training steps)
                    or a str corresponding to the checkpoint name/filename
        """
        checks = [(f[0], f[1].stem, f[1].name) for f in self.get_checkpoints()]
        checkpoint_id = None
        for check in checks:
            if checkpoint in check:
                if checkpoint_id is not None:
                    raise ValueError("duplicate matches for checkpoint found" +
                                     f"; {checkpoint} -> {checkpoint_id} " +
                                     f"and {checkpoint} -> {check}")
                checkpoint_id = check[1]
        if checkpoint_id is None:
            raise ValueError(f"Checkpoint {checkpoint} doesn't exist")
        self.agent.load_state(checkpoint_id)
        new_exp_id = self.config.experiment_id + f"_branch_{checkpoint_id}"
        self.branch_experiment(new_exp_id)
        exp_path = Path(self.config.basepath)/new_exp_id
        # Filter test episodes after checkpoint ###############################
        steps = [(int(f.name.split('_')[3]), f)
                 for f in (exp_path/'logs').iterdir()
                 if len(f.name.split('_')) > 1]
        steps_to_delete = [s for s in steps if s[0] > self.agent.training_steps]
        for step, test_episode in steps_to_delete:
            Path(test_episode).unlink()
        train_df, test_df = self.load_logs()
        # Filter train metrics after checkpoint ###############################
        # train_df = train_df[train_df['training_steps'] < self.agent.training_steps]
        train_df = train_df.iloc[:self.agent.training_steps]
        train_df.to_hdf(self.logdir/'train.hdf', key='train', mode='w')
        # Filter test metrics after checkpoint ################################
        test_df = test_df[test_df['training_steps'] < self.agent.training_steps]
        test_df.to_hdf(self.logdir/'test.hdf', key='run_history', mode='w')

    def get_checkpoints(self):
        checks = [(int(f.stem.split('_')[1]), f)
                  for f in self.agent.savepath.iterdir()
                  if len(f.name.split('_')) > 1]
        checks.sort(reverse=True)
        return checks

    def branch_experiment(self, new_exp_id: str):
        if self.config.experiment_id == new_exp_id:
            raise ValueError("branch name must be different from parent")
        old_exp_id = self.config.experiment_id
        old_exp_path = Path(self.config.basepath)/old_exp_id
        new_exp_path = Path(self.config.basepath)/new_exp_id
        new_exp_path.mkdir()
        self.config.experiment_id = new_exp_id
        # Copy over logs and models from parent experiment
        for dirr in ('logs', 'models'):
            (new_exp_path/dirr).mkdir()
            for _file in (old_exp_path/dirr).iterdir():
                shutil.copy(_file, new_exp_path/dirr/_file.name)
        self.config.save()  # automatically saves to experiment path
        replay_buff_path = old_exp_path/'replay_buffer.pkl'
        if replay_buff_path.is_file():
            shutil.copy(replay_buff_path, new_exp_path/'replay_buffer.pkl')
        self = Trainer.from_config(self.config)  # automatically do init

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
        nsteps = nsteps or self.train_steps
        train_metrics = []
        test_metrics = []
        i = self.agent.training_steps
        steps_since_test = 0
        steps_since_save = 0
        steps_since_flush = 0

        # fill replay buffer up to replay_min_size before training
        self.logger.info('initializing buffer')
        self.agent.initialize_buffer()
        test_metrics.append((i, self.agent.test_episode()))

        self.logger.info("Starting Training")
        train_loop = self.agent.step(nsteps, log_freq=2000)

        progress_bar = tqdm(total=i+nsteps, colour='#9fc693')
        progress_bar.update(i)
        # max_running_reward = 0.
        try:
            while True:
                train_metric = next(train_loop)
                train_metrics.extend(train_metric)
                training_steps_taken = self.agent.training_steps - i
                progress_bar.update(training_steps_taken)
                i = self.agent.training_steps
                # max_running_reward = max(max_running_reward,
                #                          train_metric[-1]['running_reward'])

                if steps_since_test >= self.test_freq:
                    self.logger.info("Testing Model")
                    test_metrics.append((i, self.agent.test_episode()))
                    steps_since_test = 0

                if steps_since_save >= self.config.model_save_freq:
                    self.logger.info("Saving Agent State")
                    self.agent.checkpoint()  # checkpoint history
                    self.agent.save_state()  # main lineage
                    steps_since_save = 0

                if steps_since_flush >= self.log_freq:
                    self.logger.info("Saving Log Metrics")
                    self.save_logs(train_metrics, test_metrics)
                    train_metrics, test_metrics = [], []
                    steps_since_flush = 0

                steps_since_test += training_steps_taken
                steps_since_save += training_steps_taken
                steps_since_flush += training_steps_taken

                if self.terminate_early:  # set from external thread
                    raise StopIteration

        except StopIteration:
            if self.terminate_early:
                print('Training terminated early')
            else:
                print('Done training')
            self.terminate_early = False

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
            self.logger.info("saving buffer")
            self.agent.save_buffer()
            train_metrics, test_metrics = self.load_logs()
            progress_bar.close()
            return train_metrics, test_metrics


    def run_server(self, port: int = None):
        while True:
            msg = self.socket.recv_json()



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



