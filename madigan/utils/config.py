from typing import List
from pathlib import Path
import json
import yaml
import numpy as np


class Config(dict):
    """
    Wraps a dictionary to have its keys accesible like attributes
    I.e can do both config['steps'] and config.steps to get/set items


    Note - Recursively applies this class wrapper to each dictionary value in the parent dict
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        recursive = True
        if recursive:
            for k, v in self.items():
                if isinstance(v, dict):
                    self[k] = Config(v)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        raise AttributeError("No such attribute: " + name)

    @classmethod
    def from_path(cls, filepath):
        """ Loads config from given filepath """
        return cls(load_config(Path(filepath)))

    @classmethod
    def from_exp_path(cls, exp_path):
        """ Loads config from experiment basepath- assuming filename is config.yaml"""
        config_path = Path(exp_path) / 'config.yaml'
        return cls(load_config(config_path))

    def save(self, path=None):
        """ Saves to experiment basepath as well as additional local path if provided"""
        if path is not None:
            path = Path(path)
            if not path.parent.is_dir():
                path.mkdir(parents=True)
            save_config(self, path)
        exp_path = Path(self.basepath)/self.experiment_id
        if not exp_path.is_dir():
            exp_path.mkdir(parents=True)
        save_config(self, exp_path / "config.yaml")

    def to_dict(self):
        return config_to_dict(self)


def load_config_json(path):
    with open(path, 'r') as f:
        out = json.load(f)
    return Config(out)


def save_config_json(config, path, write_mode='w'):
    with open(path, write_mode) as f:
        json.dump(dict(config), f)


def load_config(path):
    with open(path, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return Config(conf)


def config_to_dict(config: Config):
    "recursively converts config objects to dict"
    config = dict(config)
    for k, v in config.items():
        if isinstance(v, Config):
            config[k] = config_to_dict(v)
    return config


def save_config(config, path, write_mode='w'):
    config = config_to_dict(config)
    with open(path, write_mode) as f:
        yaml.dump(config, f)


def make_config(
        ########################################################################
        # GLOBAL ###############################################################
        experiment_id: str = "",  # Unique ID for each experiment
        basepath: str = "/media/hemu/Data/madigan/experiments",
        parent_id: str = "",
        # experiment_path = basepath + /experiment_id - gets implicitly created
        ########################################################################
        # ENV ##################################################################
        env_type: str = "Synth",
        data_source_type: str = "Synth",
        data_source_config: dict = None,  # If a training/testing environment,
        # then settings are needed for env data
        init_cash: int = 1_000_000,
        required_margin: float = 1.,  # proportion of transaction size
        maintenance_margin: float = 0.25,  # proportion of pnl
        transaction_cost_abs: float = 0.,  # absolute
        transaction_cost_rel: float = 0.,  # proportion of transcation_size
        slippage_abs: float = 0.,  # absolute amount
        slippage_rel: float = 0.,  # proportion of price level
        assets: List[str] = None,

        #######################################################################
        # Preprocessor ########################################################
        preprocessor_type="WindowedStacker",
        preprocessor_config: dict = None,

        # REPLAY BUFFER #######################################################
        replay_size: int = 100000,  # Size of replay buffer
        replay_min_size: int = 50_000,  # Min size before training
        episode_length: int = 1024,
        nstep_return: int = 3,  # Agent/Model spec

        # Training ############################################################
        nsteps: int = 100000,  # number_of_training_steps to run for
        train_steps: int = 100_000,
        train_freq: int = 4,  # Train every k steps
        log_freq: int = 10000,  # Log data/results at this freq
        test_freq: int = 32000,  # Run test episodes at this freq
        target_update_freq: int = 32000,  # Update offline target at this freq
        model_save_freq: int = 64000,  # Save models at this freq
        reward_clip: tuple = (-1., 1.),

        # Testing  ############################################################
        test_steps: int = 1024,  # number of testing steps to run each 'episode' for

        #######################################################################
        # AGENT ###############################################################
        agent_type="DQN",  # String of class name
        double_dqn=False,  # Agent/Model spec
        dueling=False,  # Agent/Model spec
        iqn=False,  # Agent/Model spec
        nTau1=32,
        nTau2=32,
        tau_embed_size=64,
        k_huber=1.,
        discount=0.99,  # Agent/Model spec
        tau_soft_update=1e-4,
        expl_eps=1.,  # Initial eps if eps-greedy  is used for exploration
        expl_eps_min=0.1,
        expl_eps_decay=1e-6,
        expl_noise_sd=0.1,
        batch_size=32,
        greedy_eps_testing=0.,
        discrete_actions=True,  # Agent/Model spec
        discrete_action_atoms=11,  # Agent/Model spec
        lot_unit_value=1_000,  # Agent parameter
        unit_size_proportion_avM=0.1,
        transaction_thresh=0.02,
        proximal_portfolio_penalty=0.5,

        # MODEL ###############################################################
        model_class="ConvNet",  # String of model class
        critic_model_class="ConvCriticQ",  # String of model class
        actor_model_class="ConvPolicyDeterministic",  # String of model class
        d_model=256,  # dimensionality of model
        d_model_project=256,  # dimensionality of model
        n_layers=4,  # number of layer units
        n_feats=1,  # 1 corresponds to an input of just price
        lr=1e-3,  # learning rate
        lr_critic=1e-3,  # learning rate
        lr_actor=1e-4,  # learning rate
        optim_eps=1e-8,  # eps - parameter for torch.optim
        momentum=0.9,  # parameter for torch.optim
        betas=(0.9, 0.999),  # parameter for torch.optim
        optim_wd=0,  # parameter for torch.optim
):
    assert experiment_id != "", "must specify experiment id"
    assert assets is not None, "Must specify list of asset names/codes"
    model_config = {
        'model_class': model_class,
        'critic_model_class': critic_model_class,
        'actor_model_class': actor_model_class,
        'd_model': d_model,
        'd_model_project': d_model_project,
        'n_layers': n_layers,
        'action_atoms': discrete_action_atoms,
        'dueling': dueling,
        'iqn': iqn,
        'nTau': nTau1,
        'nTau1': nTau1,
        'nTau2': nTau2,
        'tau_embed_size': tau_embed_size,
        'discrete_actions': discrete_actions,
        'discrete_action_atoms': discrete_action_atoms,
        'lot_unit_value': lot_unit_value,
    }
    optim_config = {
        'type': 'Adam',
        'lr': lr,
        'lr_critic': lr_critic,
        'lr_actor': lr_actor,
        'eps': optim_eps,
        'momentum': momentum,
        'betas': betas,
        'weight_decay': optim_wd,
    }
    agent_config = {
        'type': agent_type,
        'nsteps': nsteps,
        'replay_size': replay_size,
        'episode_length': episode_length,
        'replay_min_size': replay_min_size,
        'train_freq': train_freq,
        'target_update_freq': target_update_freq,
        'batch_size': batch_size,
        'discrete_action_atoms': discrete_action_atoms,
        'double_dqn': double_dqn,
        'dueling': dueling,
        'iqn': iqn,
        'nTau': nTau1,
        'nTau1': nTau1,
        'nTau2': nTau2,
        'k_huber': k_huber,
        'tau_embed_size': tau_embed_size,
        'discount': discount,
        'nstep_return': nstep_return,
        'action_atoms': discrete_action_atoms,
        'tau_soft_update': tau_soft_update,
        'greedy_eps_testing': greedy_eps_testing,
        'eps': expl_eps,
        'eps_min': expl_eps_min,
        'eps_decay': expl_eps_decay,
        'expl_noise_sd': expl_noise_sd,
        'reward_clip': reward_clip,
        'transaction_thresh': transaction_thresh,
        'unit_size_proportion_avM': unit_size_proportion_avM,
        'proximal_portfolio_penalty': proximal_portfolio_penalty,
    }
    config = dict(
        basepath=basepath,
        experiment_id=experiment_id,
        parent_id=parent_id,
        # overwrite_exp=overwrite_exp,
        transaction_cost_abs=transaction_cost_abs,
        transaction_cost_rel=transaction_cost_rel,
        slippage_abs=slippage_abs,
        slippage_rel=slippage_rel,
        env_type=env_type,
        init_cash=init_cash,
        required_margin=required_margin,
        maintenance_margin=maintenance_margin,
        assets=assets,
        lot_unit_value=lot_unit_value,
        n_assets=len(assets),
        discrete_actions=discrete_actions,
        discrete_action_atoms=discrete_action_atoms,
        data_source_type=data_source_type,
        data_source_config=data_source_config,
        preprocessor_type=preprocessor_type,
        preprocessor_config=preprocessor_config,
        agent_type=agent_type,
        agent_config=agent_config,
        model_config=model_config,
        optim_config=optim_config,
        train_steps=train_steps,
        reward_clip=reward_clip,
        test_steps=test_steps,
        test_freq=test_freq,
        log_freq=log_freq,
        model_save_freq=model_save_freq,
        # min_tf=window_length,
    )
    return Config(config)


default_config = make_config(experiment_id="default_config",
                             assets=[str(i) for i in range(4)])
