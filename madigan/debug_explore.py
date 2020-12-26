import os
import signal
import sys
import ast
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from madigan.utils.config import load_config

def exit():
    """
    Useful for exiting interactive process properly.
    Otherwise process is just 'Stopped' but stays alive
    """
    os.kill(os.getpid(), signal.SIGKILL)


def load_logs(logpath):
    array_cols = ('action', 'transaction', 'transactionCost', 'transactionUnit',
                    'prices', 'ledger', 'ledgerNormed')
    df = pd.read_csv(debug_path, converters={col: ast.literal_eval for col in array_cols})
    metrics = {}
    for col in df.columns:
        if col in array_cols:
            metrics[col] = np.array(df[col].tolist())
        else:
            metrics[col] = df[col]
    metrics['running_reward_pre_recon'] = np.cumsum(metrics['reward_pre_shape'])
    metrics['running_reward_post_recon'] = np.cumsum(metrics['reward_post_shape'])
    eq = metrics['equity'].values
    eq_diff = np.log(eq[1:] / eq[:-1] )

    metrics['running_reward_equity_recon'] = np.cumsum(eq_diff)

    return metrics

def plot_a(m):
    fig, axes = plt.subplots(4, 4, sharex='all')
    for ax, label in zip(axes.flatten(),
        ('running_reward', 'reward_pre_shape', 'reward_post_shape', 'running_reward_pre_recon',
        'running_reward_post_recon', 'running_reward_equity_recon', 'usedMargin', 'equity', 
        'pnl', 'transactionCost', 'transactionUnit', 'cash',
        'availableMargin', 'borrowedAssetValue', 'ledger', 'usedMargin')):
        ax.plot(m[label], label=label)
        ax.set_title(label)
    return fig, axes
    
def plot_b(m):
    fig, axes = plt.subplots(3, 4, sharex='all')
    for ax, label in zip(axes.flatten(),
        ('running_reward','running_reward_pre_recon', 'reward_pre_shape', 'reward_post_shape',
        'equity', 'transactionCost', 'transactionUnit', 'cash',
        'availableMargin', 'borrowedAssetValue', 'ledger', 'ledgerNormed')):
        ax.plot(m[label], label=label)
        ax.set_title(label)
    return fig, axes

if __name__ == "__main__":
    if len(sys.argv)>1:
        config_file = sys.argv[1]
    else:
        config_file = 'config.yaml'

    config = load_config(config_file)
    debug_path = Path(config.basepath)/config.experiment_id/'logs/debug_trainloop.csv'
    m = load_logs(debug_path)
    fig1, _ = plot_a(m)
    fig2, _ = plot_b(m)
    

