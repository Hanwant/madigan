from .cpp.build import Env, Assets
import numpy as np
# from .synth import Synth

def make_env(config):
    if config.env_type in ("Synth", ):
        assets = Assets(config.assets)
        if config.data_source_config is not None:
            env = Env(config.data_source_type, assets, config.init_cash, config)
        else:
            env = Env(config.data_source_type, assets, config.init_cash)
        env.setRequiredMargin(config.required_margin)
        env.setMaintenanceMargin(config.maintenance_margin)
        env.setTransactionCost(config.transaction_cost_rel, config.transaction_cost_rel)
        env.setSlippage(config.slippage_rel, config.slippage_rel)
        return env
    raise NotImplementedError(f"Env type {config.env_type} not implemented")

def get_env_info(env):
    return {
        "riskInfo": env.checkRisk(),
        "prices": np.array(env.currentPrices, copy=True),
        "equity": env.equity,
        "cash": env.cash,
        "pnl": env.pnl,
        "balance": env.portfolio.balance,
        "availableMargin": env.availableMargin,
        "usedMargin": env.usedMargin,
        "borrowedAssetValue": env.borrowedAssetValue,
        "ledger": np.array(env.ledger, copy=True),
        "ledgerNormed": np.array(env.ledgerNormed, copy=True),
    }
