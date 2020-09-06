from random import random
import logging
from tqdm import tqdm
import numpy as np
from ..utils import SARSD, State, ReplayBuffer
from ..fleet import make_agent, DQN
from ..environments import make_env, Synth
from .test import test


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

    for i in iterator:

        if random() < eps:
            action = env.action_space.sample()
        else:
            action = agent(state)
        eps *= (1-eps_decay)

        next_state, reward, done, info = env.step(action)
        rb.add(SARSD(state, action, reward, next_state, done))
        state = next_state

        train_metric = None
        test_metric = None

        if done:
            # print('Done', end='\r', flush=True)
            # print(info['Event'], end='\r', flush=True)
            state = env.reset()

        if len(rb) >= config.min_rb_size:
            if steps_since_train > config.train_freq:
                data = rb.sample(config.batch_size)
                train_metric = agent.train_step(data)
                train_metric['rewards'] = data.reward
                # returns = res['G_t'].mean().item()
                # train_metric = {'returns': returns, 'loss': res['loss']}
                steps_since_train = 0
            if steps_since_target_update > config.test_freq:
                test_metric = test(agent, env, config.test_steps)
                # returns = sum(res['returns'])
                # test_metric = {'returns': returns}
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


class Trainer:
    def __init__(self, agent, env, config, print_progress=True):
        self.agent = agent
        self.env = env
        self.config = config
        self.print_progress = print_progress
        self = iter(self)

    @classmethod
    def from_config(cls, config, print_progress=True):
        agent = make_agent(config)
        env = make_env(config)
        return cls(agent, env, config, print_progress=print_progress)

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
        try:
            while True:
                train_metric, test_metric = next(self.train_loop)
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
            import ipdb; ipdb.set_trace()

        finally:
            if len(train_metrics) > 0:
                train_metrics = {k: [train_metric[k] for train_metric in train_metrics] for k in train_metrics[0].keys()}
            if len(test_metrics) > 0:
                test_metrics = {k: [test_metric[k] for test_metric in test_metrics] for k in test_metrics[0].keys()}
            else:
                test_metrics = test(self.agent, self.env, nsteps=self.config.test_steps)
            return train_metrics, test_metrics








