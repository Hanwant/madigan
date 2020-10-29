from typing import Union
from pathlib import Path
import pandas as pd

def test_summary(test_metrics):
    df = pd.DataFrame(test_metrics)
    out = {'mean_equity': df['equity'].mean(),
            'final_equity': df['equity'].iloc[-1],
            'mean_reward': df['reward'].mean(),
            'mean_transaction_cost': df['transaction_cost'].mean(),
            'total_transaction_cost': df['transaction_cost'].sum()
           }
            # 'mean_qvals': df['qvals'].mean()}
    return out

def save_to_hdf(path: Union[str, Path], key: str, df: pd.DataFrame, append_if_exists: bool=True):
    """
    Generic for saving df to hdf5 (and appending by default - useful for
    fast i/o of logs)
    """
    with pd.HDFStore(path, mode='a') as f:
        f.put(key, df, format='t', append=append_if_exists)

def load_from_hdf(path: Union[str, Path], key: str):
    """
    Generic for loading df from hdf5 (useful for fast i/o of logs)
    """
    with pd.HDFStore(path, mode='r') as f:
        if '/'+key in f.keys():
            return pd.read_hdf(f, key=key)
    return None
