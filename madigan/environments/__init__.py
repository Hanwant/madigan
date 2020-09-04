from .synth import Synth

def make_env(config):
    if config.env_type == "Synth":
        return Synth(**config)
    else:
        raise NotImplementedError(f"Env type {config.env_type} not implemented")
