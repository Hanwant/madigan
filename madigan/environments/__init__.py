from .cpp.build import Env, Assets
# from .synth import Synth

def make_env(config):
    if config.env_type in ("Synth", ):
        assets = Assets(config.assets)
        if config.generator_params is not None:
            env = Env(config.data_source_type, assets, config.init_cash, config)
        else:
            env = Env(config.data_source_type, assets, config.init_cash)
        env.setRequiredMargin(config.required_margin)
        env.setMaintenanceMargin(config.maintenance_margin)
        env.setTransactionCost(config.transaction_cost_rel, config.transaction_cost_rel)
        env.setSlippage(config.slippage_rel, config.slippage_rel)
        return env
    raise NotImplementedError(f"Env type {config.env_type} not implemented")
