from typing import Union
from pathlib import Path
import pandas as pd

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
