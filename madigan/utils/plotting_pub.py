"""
Utils for making figures for publication or presentation
"""
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

rc_params = {
    'text.usetex': False,
    'lines.linewidth': 2,
    'axes.labelsize': 'large'
}

sns.set_theme(context="paper", font_scale=3, rc=rc_params)


def save_fig(fig, savepath):
    """ Save all figures in pdf format - best for latex """
    savepath = Path(savepath)
    if not savepath.parent.is_dir():
        savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, format='pdf', bbox_inches='tight')

def save_figs(figs, savepath):
    for label, fig in figs.items():
        save_fig(fig, Path(savepath) / f"{label}.pdf")
        # save_fig(fig['fig'], savepath/f"{fig['fname']}.pdf")
        # save_fig(fig, savepath/f'{fig.label}.pdf')

def get_episode_id(
    experiment_id,
    basepath,
    env_steps=None,
    episode_steps=None,
):
    assert env_steps is not None or episode_steps is not None
    path = Path(basepath) / experiment_id
    runs = filter(lambda x: "episode" in str(x.name), path.iterdir())
    runs = list(map(lambda x: x.name.split('_'), runs))
    if env_steps is not None:
        runs = filter(runs, lambda x: env_steps == x[3])
    if episode_steps is not None:
        runs = filter(runs, lambda x: episode_steps == x[5])
    if len(runs) > 0:
        raise ValueError("multiple matches found for env and episode steps "
                         f"combination: {env_steps}, {episode_steps}")
    return ''.join(runs[0])


def format_xticks(ticks):
    out = []
    for tick in ticks:
        tick = float(tick)
        if tick < 1000:
            out.append(str(int(tick)))
        else:
            out.append(str(int(tick / 1000)) + 'K')
    return out


def load_train_data(experiment_id, basepath):
    path = Path(basepath) / experiment_id / 'logs/train.hdf5'
    df = pd.read_hdf(path, key='train')
    return df


def load_test_history_data(experiment_id, basepath):
    path = Path(basepath) / experiment_id / 'logs/test.hdf5'
    df = pd.read_hdf(path, key='run_history')
    return df


def load_test_run_data(episode_id, experiment_id, basepath):
    path = Path(basepath) / experiment_id / f'logs/{episode_id}.hdf5'
    df = pd.read_hdf(path, key='run_history')
    return df


def make_train_figs(experiment_id, basepath):
    df = load_train_data(experiment_id, basepath).reset_index(drop=True)
    figs = {}
    for label in df.columns:
        fig, ax = plt.subplots(1, 1)
        ax.plot(df[label], label=label)
        ax.legend()
        ax.set_xlabel('training_step')
        # figs.append(fig)
        # figs.append({'fig': fig, 'fname': label})
        figs[label] = fig
    figs['running_reward'].axes[0].set_ylabel('cumulative rewards')
    return figs


def make_labels_from_test_df(df):
    """
    Formats column names in df.
    Returns labels dict, along with list of metrics aggregation timeframes
    and assets
    """
    timeframes = [
        lab[22:] for lab in df.keys() if 'equity_returns_offset_' in lab
    ]
    assets = [lab[18:] for lab in df.keys() if 'time_spent_in_pos' in lab]

    timeframe_labels = {  # nicer formatting than default column names
        **{
            f'equity_returns_offset_{tf}': f'mean returns (offset {tf})'
            for tf in timeframes
        },
        **{
            f'equity_sharpe_offset_{tf}': f'sharpe (offset {tf})'
            for tf in timeframes
        },
        **{
            f'equity_sortino_offset_{tf}': f'sortino (offset {tf})'
            for tf in timeframes
        }
    }

    time_spent_labels = {
        f'time_spent_in_pos_{asset}': f'time in positions ({asset})'
        for asset in assets
    }

    labels = {'mean_reward': 'mean reward', 'max_drawdown': 'max drawdown'}
    labels.update(timeframe_labels)
    labels.update(time_spent_labels)
    return labels, timeframes, assets


def _make_test_history_plot_old(ax, key, label, ylabel, x, dat, smooth_window,
                            sd_window, colour):

    if smooth_window is not None:
        dat = dat.rolling(smooth_window, min_periods=1).mean()
    if sd_window is not None:
        sd = dat.rolling(sd_window, min_periods=1).std()
        upper = dat + sd
        lower = dat - sd
        if 'time_spent' in key:
            upper = np.clip(upper, None, 1.)
    line = ax.plot(x, dat, label=label, color=colour)
    if sd_window is not None:
        colour = line[-1].get_color()
        ax.fill_between(x, lower, upper, alpha=0.25, color=colour)
        if ((np.nanmax(dat) - np.nanmin(dat)) / (np.nanmin(dat) + 1.)) > 1e7:
            ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('training step')
    return ax

def _make_test_history_plot(ax, key, line_label, y_label, x, dat, smooth_window,
                            sd_window, colour):

    idx = (np.arange(len(dat)) / 5).astype(np.int)
    if smooth_window is not None:
        # _dat = dat.rolling(smooth_window, min_periods=1).mean()
        _dat = dat.groupby(idx).mean()
        x = x[0::5]
    else:
        _dat = dat
    if sd_window is not None:
        # assert smooth_window is not None
        if smooth_window is None:
            sd = dat.rolling(sd_window, min_periods=1).std()
        else:
            sd = dat.groupby(idx).std()
        upper = _dat + sd
        lower = _dat - sd
        if any([lab in key for lab in ('time_spent', 'drawdown')]):
            upper = np.clip(upper, None, 1.)
            lower = np.clip(lower, 0., None)
    line = ax.plot(x, _dat, label=line_label, color=colour)
    if sd_window is not None:
        colour = line[-1].get_color()
        ax.fill_between(x, lower, upper, alpha=0.25, color=colour)
        if ((np.nanmax(_dat) - np.nanmin(_dat)) / (np.nanmin(_dat) + 1.)) > 1e7:
            ax.set_yscale('log')
    ax.set_ylabel(y_label)
    ax.set_xlabel('training step')
    return ax

def make_test_history_figs(
    df,
    smooth_window: int = None,
    sd_window: int = None,
    # groups: list = None,
    colour=None,
    figs: dict = None,
):
    """
    df: pandas dataframe loaded from file
    labels
    smooth_window = window over which to aggregate mean for each plot
                    default = None for no aggregation
    sd_window = window over which to aggregate sd to get confidence band
                default = None for no confidence bands
    # groups = list of lists
    Parameters for Comparison Use Case:
        colour = colour for plots
        figs = dict of figs on which to add plots - for comparing
               allows one function for both use cases. defualt = None
    """
    labels, timeframes, assets = make_labels_from_test_df(df)
    found = [label in df.keys() for label in labels]
    assert all(
        found
    ), f"Missing {[label for label in labels if label not in df.keys()]}"
    if figs is None:
        figs = {}
    x = df['training_steps']
    for key, label in labels.items():
        if key not in figs:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        else:
            fig, ax = figs[key], figs[key].axes[0]
        ax = _make_test_history_plot(ax, key, label, label, x, df[key],
                                     smooth_window, sd_window, colour)
        figs[key] = fig
    return figs


def make_test_history_grouped_figs(
    df,
    label_groups,
    smooth_window: int = None,
    sd_window: int = None,
    # groups: list = None,
    colour=None,
    figs: dict = None,
):
    """
    df: pandas dataframe loaded from file
    label_groups: dict[str: dict[str: str]] = groups of variable keys and thier labels
    smooth_window = window over which to aggregate mean for each plot
                    default = None for no aggregation
    sd_window = window over which to aggregate sd to get confidence band
                default = None for no confidence bands
    # groups = list of lists
    Parameters for Comparison Use Case:
        colour = colour for plots
        figs = dict of figs on which to add plots - for comparing
               allows one function for both use cases. defualt = None
    """
    if figs is None:
        figs = {}
    x = df['training_steps']
    for group, labels in label_groups.items():
        if group not in figs:
            fig, axes = plt.subplots(1,
                                     len(labels),
                                     figsize=(10, 5),
                                     squeeze=False)
            axes = axes[0]
        else:
            fig = figs[group]
            axes = fig.axes
        for i, (key, label) in enumerate(labels.items()):
            ax = axes[i]
            _make_test_history_plot(ax, key, label, label, x, df[key],
                                    smooth_window, sd_window, colour)
        figs[group] = fig
    return figs


def make_comparative_test_figs(exp_ids: dict,
                               basepath,
                               smooth_window: int = None,
                               sd_window: int = None):
    """ exp_ids: dict of experiment id: nicer experiment name for titles """
    colours = sns.color_palette('dark')
    assert len(colours) > len(
        exp_ids
    ), f"# of exp {len(exp_ids)} > number of colours {len(colours)} " + \
    "too many exp comparisons!"

    figs = {}
    for i, exp in enumerate(exp_ids):
        df = load_test_history_data(exp, basepath)
        figs = make_test_history_figs(df,
                                      smooth_window,
                                      sd_window,
                                      colour=colours[i],
                                      figs=figs)
    legend_lines = [
        Line2D([0], [0], color=colours[i]) for i in range(len(exp_ids))
    ]
    fig, ax = plt.subplots()
    fig.legend(legend_lines, exp_ids.values())
    figs['legend'] = fig
    ax.remove()
    return figs


def make_multi_asset_plot(ax, x, ys, assets=None):
    n_assets = ys.shape[1]
    assets = assets if assets is not None else [
        f"asset_{i}" for i in range(n_assets)
    ]
    if n_assets != len(assets):
        raise ValueError("data shape does not match number of assets provided")
    for i, asset in enumerate(assets):
        ax.plot(x, ys[:, i], label=asset)
    if n_assets > 1:
        ax.legend(loc='upper left')
    return ax


def make_figs(data: Union[dict, pd.DataFrame], assets=None, x_key=None):
    if x_key is None:
        x = None
    else:
        x = data[x_key]
    figs = {}
    for label in data.keys():
        if x is None or len(x) != len(data[label]):
            x = range(len(data[label]))
        try:
            # dat = data[label].squeeze()
            dat = data[label]
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            if len(dat.shape) == 1:
                ax.plot(x, dat, label=label)
                # legend only for multi asset plots
                # ax.legend(loc='upper left')
                if ((np.nanmax(dat) - np.nanmin(dat)) / np.nanmin(dat)) > 1e7:
                    ax.set_yscale('log')
            elif len(dat.shape) == 2:
                ax = make_multi_asset_plot(
                    ax,
                    x,
                    dat,
                    assets=assets,
                )
            else:
                ax.imshow(dat.squeeze())  # vmin=0., vmax=1.) #, cmap='gray'
            ax.set_ylabel(label)
            ax.set_xlabel('step')
            figs[label] = fig
        except Exception as E:
            print(E)
            print('skipping: ', label)
    return figs


