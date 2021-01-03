"""
Utils for making figures for publication or presentation
"""
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def save_fig(fig, savepath):
    """ Save all figures in pdf format - best for latex """
    fig.save(savepath, format='pdf')

def get_episode_id(experiment_id,
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


def make_train_figs(experiment_id, basepath, show=True, save=True,
                    savepath=None):
    df = load_train_data(experiment_id, basepath)
    savepath = savepath or Path(os.getcwd())/experiment_id
    figs = []
    for label in df.columns:
        fig, ax = plt.subplot(1, 1, figsize=(15, 10))
        ax.plot(df[label], label=label)
        ax.legend()
        figs.append({'fig': fig, 'fname': label})
    return figs


def save_figs(figs, savepath):
    for fig in figs:
        save_fig(fig['fig'], savepath/f"{fig['fname']}.pdf")
