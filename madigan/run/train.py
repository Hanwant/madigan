from random import random
from tqdm import tqdm
import numpy as np
from ..utils import SARSD, State, ReplayBuffer
from ..fleet import make_agent, DQN
from ..environments import make_env, Synth
from .test import test


def train_loop(agent, env, config):

    rb = ReplayBuffer(config.rb_size)

    eps = config.expl_eps
    eps_min = config.expl_eps_min
    eps_decay = config.expl_eps_decay

    state = env.reset()
    i = agent.total_steps
    tq = tqdm(total=config.nsteps)
    nsteps = i + config.nsteps
    eps *= (1-eps_decay)**i
    local_steps = 0
    steps_since_target_update = 0
    steps_since_train = 0

    import time
    for i in range(i, nsteps):
        tq.update(1)

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
                res = agent.train_step(data)
                returns = res['G_t'].mean()
                loss = np.mean(res['loss'])
                train_metric = {'returns': returns, 'loss': loss}
                steps_since_train = 0
            if steps_since_target_update > config.test_freq:
                res = test(agent, env, 1000)
                returns = sum(res['returns'])
                test_metric = {'returns': returns}
                agent.model_t.load_state_dict(agent.model_b.state_dict())
                steps_since_target_update = 0
            steps_since_train += 1
            steps_since_target_update += 1
        local_steps += 1
        yield train_metric, test_metric

def train_gen(agent, env, config):
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
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config
        self.train_loop = iter(train_loop(agent, env, config))

    @classmethod
    def from_config(cls, config):
        agent = make_agent(config)
        env = make_env(env)
        return cls(agent, env, config)


    def __iter__(self):
        return self

    def __next__(self):
        """
        Returns a tuple:
        (train_metric, test_metric)
        check for None
        """
        return next(self.train_loop)

    def train(self):
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

        return train_metrics, test_metrics








