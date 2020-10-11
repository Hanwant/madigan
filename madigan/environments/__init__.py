from .cpp import Env, Assets
from .synth import Synth

def make_env(config):
    if config.env_type == "Synth":
        g_params = config.generator_params if config.generator_params is not None else None
        assets = Assets(config.assets)
        if g_params is not None:
            env = Env("Synth", assets, config.init_cash, config)
        else:
            env = Env("Synth", assets, config.init_cash)
        env.setRequiredMargin(config.required_margin)
        env.setMaintenanceMargin(config.maintenance_margin)
        return env
    else:
        raise NotImplementedError(f"Env type {config.env_type} not implemented")

