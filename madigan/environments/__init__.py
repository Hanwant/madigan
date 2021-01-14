from .cpp.build import Env, Assets
import numpy as np
# from .synth import Synth

def make_env(config):
    if config.env_type in ("Synth", ):
        if config.data_source_config is not None:
            env = Env(config.data_source_type, config.init_cash, config)
        else:
            env = Env(config.data_source_type, config.init_cash)
        env.setRequiredMargin(config.required_margin)
        env.setMaintenanceMargin(config.maintenance_margin)
        env.setTransactionCost(config.transaction_cost_rel,
                               config.transaction_cost_abs)
        env.setSlippage(config.slippage_rel, config.slippage_abs)
        return env
    raise NotImplementedError(f"Env type {config.env_type} not implemented")

def get_env_info(env):
    return {
        "timestamp": env.timestamp,
        "riskInfo": env.checkRisk(),
        "prices": np.array(env.currentPrices, copy=True),
        "equity": env.equity,
        "cash": env.cash,
        "pnl": env.pnl,
        "balance": env.portfolio.balance,
        "availableMargin": env.availableMargin,
        "usedMargin": env.usedMargin,
        "borrowedAssetValue": env.borrowedAssetValue,
        "borrowedMargin": env.borrowedMargin,
        "ledger": np.array(env.ledger, copy=True),
        "ledgerNormed": np.array(env.ledgerNormed, copy=True),
    }

def format_env_info(info_dict):
    """ for printing """
    # return yaml.dump(info_dict, default_flow_style=None, sort_keys=False)
    formatted = ""
    for k, v in info_dict.items():
        formatted += k + ": " + repr(v) + "\n"
    return formatted
