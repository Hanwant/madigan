from typing import Union

import numpy as np
import pandas as pd
import torch


def list_2_dict(list_of_dicts: list) -> dict:
    """
    aggregates a list of dicts (all with same keys) into a dict of lists

    the train_loop generator yield dictionaries of metrics at each iteration.
    this allows the loop to be interoperable in different scenarios
    The expense of getting a dict (instead of directly appending to list)
    is probably not too much but come back and profile


    """
    if isinstance(list_of_dicts, dict):
        return list_of_dicts
    if list_of_dicts is not None and len(list_of_dicts) > 0:
        if isinstance(list_of_dicts[0], dict):
            dict_of_lists = {k: [metric[k] for metric in list_of_dicts]
                             for k in list_of_dicts[0].keys()}
            return dict_of_lists
        raise ValueError('list passed to list_2_dict does not contain dicts')
    return {}


def reduce_train_metrics(metrics: Union[dict, pd.DataFrame],
                         columns: list) -> Union[dict, pd.DataFrame]:
    """
    Takes dict (I.e from list_2_dict) or pandas df
    returns dict/df depending on input type
    """
    if metrics is not None and len(metrics) > 0:
        _metrics = type(metrics)()  # Create an empty dict or pd df
        for col in metrics.keys():
            if col in columns:
                if isinstance(metrics[col][0], (np.ndarray, torch.Tensor)):
                    _metrics[col] = [m.mean().item() for m in metrics[col]]
                elif isinstance(metrics[col][0], list):
                    _metrics[col] = [np.mean(m).item() for m in metrics[col]]
                else:
                    _metrics[col] = metrics[col]
            else:
                _metrics[col] = metrics[col]
    else:
        return metrics
    return _metrics


def test_summary(test_metrics: Union[dict, pd.DataFrame]) -> pd.DataFrame:
    """
    Given metrics from a test episode, returns a summary.
    Needs to be updated to include backtest style stats
    I.e drawdown, sharpe etc - sharpe needs timestamps
    """
    df = pd.DataFrame(test_metrics)
    qvals = {}
    for qval in ('qvals', 'qvals1', 'qvals2'):
        if qval in df.columns:
            qvals['mean_'+qval] = np.array(df[qval].tolist()).mean()

    out = {'mean_equity': df['equity'].mean(),
           'final_equity': df['equity'].iloc[-1],
           'mean_reward': df['reward'].mean(),
           'mean_transaction_cost': np.array(df['transaction_cost'].tolist()).mean(),
           'total_transaction_cost': np.array(df['transaction_cost'].tolist()).sum(),
           'nsteps': len(df),
           **qvals}
    return pd.DataFrame({k: [v] for k, v in out.items()})

# def reduce_test_metrics(test_metrics, cols=('returns', 'equity', 'cash', 'margin')):
#     out = []
#     # if isinstance(test_metrics, dict):
#     #     return list_2_dict(reduce_test_metrics([test_metrics], cols=cols))
#     keys = test_metrics[0].keys()
#     for m in test_metrics:
#         _m = {}
#         for k in keys:
#             if k not in cols:
#                 _m[k] = m[k]
#             else:
#                 if isinstance(m[k], (np.ndarray, torch.Tensor)):
#                     _m[k] = m[k].mean().item()
#                 elif isinstance(m[k], list):
#                     _m[k] = np.mean(m[k])
#                 else:
#                     try:
#                         _m[k] = np.mean(m[k])
#                     except Exception as E:
#                         import traceback
#                         traceback.print_exc()
#                         print("col passed to reduce_test_metrics did not contain ndarray/list/tensor")
#                         print("np.mean tried anyway and failed")
#         out.append(_m)
#     return out
