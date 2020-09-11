from pathlib import Path
import json
import numpy as np

class Config(dict):
    """
    Wraps a dictionary to have its keys accesible like attributes
    I.e can do both config['steps'] and config.steps to get/set items


    Note - Recursively applies this class wrapper to each dictionary value in the parent dict
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        recursive=True
        if recursive:
            for k, v in self.items():
                if isinstance(v, dict):
                    self[k] = Config(v)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

def load_config(path):
    with open(path, 'r') as f:
        out = json.load(f)
    return Config(out)

def save_config(obj, path, write_mode='w'):
    with open(path, write_mode) as f:
        json.dump(dict(obj), f)

def make_config(
        ##################################################################################
        # GLOBAL #########################################################################
        basepath="/media/hemu/Data/Markets/farm", # Path where experiments are stored
        experiment_id="", # Unique ID for each experiment
        parent_id="", # If branching from another experiment, ID of parent has to be specified
        overwrite_exp=False,
        discrete_actions=True, # Env/Agent/Model spec
        discrete_action_atoms=11, # Env/Agent/Model spec
        lot_unit_value=1_000, # Env -> Accounting/Broker parameter

        ###################################################################################
        # ENV #############################################################################
        env_type="Synth", # Env is an abstraction (I.e could mean training/testing env or live trading env)
        generator_params=None, # If a training/testing environment, then settings are needed for env data
        test_steps=1000, # number of testing steps to run each 'episode' for

        # REPLAY BUFFER ####################################################################
        rb_size=100000, # Size of replay buffer
        min_rb_size=50_000, # Min size before training
        nstep_return=1, # Agent/Model spec

        # Training #########################################################################
        nsteps=100000, # number_of_training_steps to run for
        train_freq=4, # Train every k steps
        log_freq=10000, # Log data/results at this freq
        test_freq=32000, # Run test episodes at this freq
        target_update_freq=32000, # Update offline target at this freq
        model_save_freq=64000, # Save models at this freq
        reward_clip=(-1., 1.),

        ####################################################################################
        # AGENT + MODEL ####################################################################

        agent_type="DQN", # String of class name
        double_dqn=False, # Agent/Model spec
        dueling=False, # Agent/Model spec
        iqn=False, # Agent/Model spec
        discount=0.99, # Agent/Model spec
        expl_eps=1., # Initial eps if eps-greedy  is used for exploration
        expl_eps_min=0.1,
        expl_eps_decay=1e-6,
        batch_size=32,
        greedy_eps_testing=0.,

        # MODEL ####################################################################
        model_class="ConvModel", # String of model class
        d_model=256, # dimensionality of model
        n_layers=4, # number of layer units
        n_assets=4, # number of assets being traded - expected size of input
        min_tf=64, # global time_frames parameter
        n_feats=1, # 1 corresponds to an input of just price
        lr=1e-3, # learning rate
        optim_eps=1e-8, # eps - parameter for torch.optim
        momentum=0.9, # parameter for torch.optim
        betas=(0.9, 0.999), # parameter for torch.optim
        optim_wd=0, # parameter for torch.optim
):
    assert experiment_id != "", "must specify experiment id"
    model_config = {
        'model_class': model_class,
        'd_model': d_model,
        'n_layers': n_layers,
        'n_feats': n_feats,
        'action_atoms': discrete_action_atoms,
        'n_assets': n_assets,
        'min_tf': min_tf,
        'dueling': dueling,
        'iqn': iqn,
    }
    optim_config = {
        'type': 'Adam',
        'lr': lr,
        'eps': optim_eps,
        'momentum': momentum,
        'betas': betas,
        'weight_decay': optim_wd,
    }
    agent_config = {
        'type': agent_type,
        'basepath': basepath,
        'model_config': model_config,
        'optim_config': optim_config,
        'double_dqn': double_dqn,
        'discount': discount,
        'nstep_return': nstep_return,
        'action_atoms': discrete_action_atoms,
        'greedy_eps_testing': greedy_eps_testing,
    }
    config = dict(
        basepath=basepath,
        experiment_id=experiment_id,
        parent_id=parent_id,
        overwrite_exp=overwrite_exp,

        env_type=env_type,
        generator_params=generator_params,
        lot_unit_value=lot_unit_value,
        n_assets=n_assets,
        discrete_actions=discrete_actions,
        discrete_action_atoms=discrete_action_atoms,

        agent_type=agent_type,
        nsteps=nsteps,
        test_steps=test_steps,
        rb_size=rb_size,
        min_rb_size=min_rb_size,
        train_freq=train_freq,
        target_update_freq=target_update_freq,
        test_freq=test_freq,
        log_freq=log_freq,
        model_save_freq=model_save_freq,

        min_tf=min_tf,
        batch_size=batch_size,
        agent_config=agent_config,
        nstep_return=nstep_return,
        expl_eps=expl_eps,
        expl_eps_min=expl_eps_min,
        expl_eps_decay=expl_eps_decay,
        reward_clip=reward_clip,
    )
    return Config(config)

