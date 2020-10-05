import math

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt


def make_grid(n):
    """
    utility function
    makes evenish 2d grid of size n
    useful for arranging plots
    """
    half = int(math.sqrt(n))
    nrows = half
    ncols = int(math.ceil(n/nrows))
    return nrows, ncols


def plot_test_metrics(data, include=('equity', 'cash', 'positions', 'margin', 'returns')):
    assert isinstance(data, (dict, pd.DataFrame)), "expected data to be a dict or pd df"
    if 'timestamp' in data.keys():
        index = pd.to_datetime(data['timestamp'])
    else:
        index = range(len(data['equity']))
    metrics = list(filter(lambda m: m in include, data.keys()))
    fig, axes = plt.subplots(*make_grid(len(metrics)), sharex=True, squeeze=False)
    ax = axes.flatten()
    for i, metric in enumerate(metrics):
        if metric in ('equity', 'cash', 'margin', 'returns'):
            ax[i].plot(index, data[metric], label=metric)
            ax[i].set_title(metric)
            ax[i].legend()
        elif metric in ('positions'):
            ax[i].imshow(np.array(data[metric])) #, cmap='gray',vmin=0., vmax=1.
            pass
    if len(ax) > len(metrics):
        extra = len(ax) - len(metrics)
        for i in range(len(ax) - extra, len(ax)):
            fig.delaxes(ax[i])
    return fig, axes




