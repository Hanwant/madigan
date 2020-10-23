import math

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def make_grid(n):
    """
    utility function
    makes evenish 2d grid of size n
    useful for arranging plots
    """
    nrows =  int(math.sqrt(n))
    ncols = int(math.ceil(n/nrows))
    return nrows, ncols


def plot_test_metrics(data, include=('prices', 'equity', 'cash', 'positions',
                                     'margin', 'returns', 'actions', 'qvals'),
                      assets=None):
    assert isinstance(data, (dict, pd.DataFrame)), "expected data to be a dict or pd df"
    if 'timestamp' in data.keys():
        index = pd.to_datetime(data['timestamp'])
    else:
        index = range(len(data['equity']))
    metrics = list(filter(lambda m: m in include, data.keys()))
    if 'prices' in metrics and (assets is None or len(assets) != len(data['prices'][0])):
        assets = ["asset_"+str(i) for i in range(len(data['prices'][0]))]
    fig, axes = plt.subplots(*make_grid(len(metrics)), sharex=True, squeeze=False)
    ax = axes.flatten()
    for i, metric in enumerate(metrics):
        if metric in ('prices', 'actions'):
            prices = np.array(data[metric])
            for j, asset in enumerate(assets):
                ax[i].plot(index, prices[:, j], label=asset)
            ax[i].legend()
            ax[i].set_title(metric)
        elif metric in ('equity', 'cash', 'margin', 'returns'):
            ax[i].plot(index, data[metric], label=metric)
            ax[i].set_title(metric)
            ax[i].legend()
        elif metric in ('positions', ):
            data_2d = np.array(data[metric]).T
            im = ax[i].imshow(data_2d) #vmin=0., vmax=1.) #, cmap='gray'
            ax[i].set_aspect(data_2d.shape[1]/data_2d.shape[0])
            ax[i].set_title(metric)
            ax[i].set_yticks(range(data_2d.shape[0]))
            ax[i].set_yticklabels(labels=assets)
            fig.colorbar(im, ax=ax[i])
        elif metric in ('qvals', ):
            assetIdx = 0
            if len(data[metric][0].shape) == 2:
                data_2d = np.stack(data[metric]).T[:, assetIdx, :]
            else:
                data_2d = np.stack(data[metric]).T
            im = ax[i].imshow(data_2d) #vmin=0., vmax=1.) #, cmap='gray'
            ax[i].set_aspect(data_2d.shape[1]/data_2d.shape[0])
            ax[i].set_title(metric)
            ax[i].set_yticks(range(data_2d.shape[0]))
            ax[i].set_yticklabels(labels=[f'action_{i}' for i in range(data_2d.shape[0])])
            fig.colorbar(im, ax=ax[i])
    if len(ax) > len(metrics):
        extra = len(ax) - len(metrics)
        for i in range(len(ax) - extra, len(ax)):
            fig.delaxes(ax[i])
    return fig, axes

def plot_train_metrics(data, include=('loss', 'td_error', 'G_t', 'Q_t',
                                      'Gt', 'Qt', 'rewards')):
    assert isinstance(data, (dict, pd.DataFrame)), "expected data to be a dict or pd df"
    if 'timestamp' in data.keys():
        index = pd.to_datetime(data['timestamp'])
    else:
        index = range(len(data['loss']))
    metrics = list(filter(lambda m: m in include, data.keys()))
    fig, axes = plt.subplots(*make_grid(len(metrics)), sharex=True, squeeze=False)
    ax = axes.flatten()
    for i, metric in enumerate(metrics):
        ax[i].plot(index, data[metric], label=metric)
        ax[i].set_title(metric)
        ax[i].legend()
    if len(ax) > len(metrics):
        extra = len(ax) - len(metrics)
        for i in range(len(ax) - extra, len(ax)):
            fig.delaxes(ax[i])
    return fig, axes


def plot_sarsd(sarsd, env_metrics=None, qvals=None):
    n_plots = 3
    n_assets = len(sarsd.state.portfolio)
    if env_metrics is not None:
        n_plots += 1
    if qvals is not None:
        n_plots += 1
    fig, axes = plt.subplots(*make_grid(n_plots), squeeze=False, figsize=(10, 7.5))
    ax = axes.flatten()
    ax[0].plot(sarsd.state.price)
    ax[0].set_title("current_state")
    ax[1].plot(sarsd.next_state.price)
    ax[1].set_title("next_state")
    current_price = sarsd.state.price[-1]
    next_price = sarsd.next_state.price[-1]
    txt = f"""
    current state port: {sarsd.state.portfolio}\n
    next state port: {sarsd.next_state.portfolio}\n
    action: {sarsd.action}
    reward: {sarsd.reward}
    price return : {(next_price-current_price)/current_price}
    done: {sarsd.done}
    """
    # risk: {sarsd.info.brokerResponse.riskInfo}
    # margin_call: {sarsd.info.brokerResponse.marginCall}
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax[2].text(0., 1., txt, transform=ax[2].transAxes, bbox=props,
               verticalalignment='top')
    ax[2].set_title("info")
    current_idx = 3
    if env_metrics is not None:
        current_idx += 1; m = env_metrics
        txt = f"""
        equity: {m['equity']}\n
        cash: {m['cash']}\n
        balance: {m['balance']}\n
        margin: {m['margin']}\n
        pnl: {m['pnl']}\n
        """
        ax[current_idx].text(0., 0., txt)
    if qvals is not None:
        current_idx += 1
        ax[current_idx].bar(range(len(qvals)), qvals)
        ax[current_idx].set_title("qvals")
    if len(ax) < axes.shape[0] * axes.shape[1]:
        extra = len(ax) - axes.shape[0] * axes.shape[1]
        for i in range(len(ax) - extra, len(ax)):
            fig.delaxes(ax[i])
    return fig, axes




