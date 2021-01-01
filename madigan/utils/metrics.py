from typing import Union

import numba as nb
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
            dict_of_lists = {
                k: [metric[k] for metric in list_of_dicts]
                for k in list_of_dicts[0].keys()
            }
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
            qvals['mean_' + qval] = np.array(df[qval].tolist()).mean()

    out = {
        'mean_equity': df['equity'].mean(),
        'final_equity': df['equity'].iloc[-1],
        'mean_reward': df['reward'].mean(),
        'mean_transaction_cost':
        np.array(df['transaction_cost'].tolist()).mean(),
        'total_transaction_cost':
        np.array(df['transaction_cost'].tolist()).sum(),
        'nsteps': len(df),
        **qvals
    }
    return pd.DataFrame({k: [v] for k, v in out.items()})


def variance(timeseries: Union[np.ndarray, pd.Series],
             timeframe: Union[int, pd.Timedelta],
             timestamps: np.ndarray = None,
             ddof: int = 1):
    """
    Computes the variance of returns of a given timeseries
    I.e price or equity curve.
    @params
    timeseries: array = the timeseries to compute returns variance for
    timeframe: window = the timeframe for return calculation.
                        (n_periods if timestamps are integer increments)
    ddof: int = degrees of freedom in vairance calculation. Default = 1.
    """
    using_dt = False
    if timestamps is None:
        if isinstance(timeseries, pd.Series):
            if isinstance(timeseries.index, pd.DatetimeIndex):
                timestamps = timeseries.index.values.view(np.int64)
                using_dt = True
            else:
                timestamps = np.arange(len(timeseries))

    arr = np.array(timeseries)
    if isinstance(timeframe, pd.Timedelta):
        timeframe = timeframe.view(np.int64)
    returns = _returns(arr, timestamps, timeframe)
    return np.nanvar(returns, ddof=ddof)


def covariance(timeseriesA: Union[np.ndarray, pd.Series],
               timeseriesB: Union[np.ndarray, pd.Series],
               timeframe: Union[int, pd.Timedelta],
               timestamps: np.ndarray = None,
               ddof: int = 1):
    """
    Computes the covariance of returns for 2 timeseries.
    I.e price vs equity curve.
    @params
    timeseriesA: array
    timeseriesB: array
    timeframe: window = the timeframe for return calculation.
                        (n_periods if timestamps are integer increments)
    ddof: int = degrees of freedom in vairance calculation. Default = 1.
    """
    assert type(timeseriesA) == type(timeseriesB)
    if timestamps is None:
        if isinstance(timeseriesA, pd.Series):
            if isinstance(timeseriesA.index, pd.DatetimeIndex):
                timestamps = timeseriesA.index.values.view(np.int64)
            else:
                timestamps = np.arange(len(timeseriesA))

    arr_1 = np.array(timeseriesA)
    arr_2 = np.array(timeseriesB)
    if isinstance(timeframe, pd.Timedelta):
        timeframe = timeframe.view(np.int64)
    ret_1 = _returns(arr_1, timestamps, timeframe)
    ret_2 = _returns(arr_2, timestamps, timeframe)
    cov = (ret_1 - ret_1.mean()) * (ret_2 - ret_2.mean())
    # cov = _expanding_covariance_of_returns(returns_1, returns_2)
    return cov.sum() / (len(cov) - ddof)


@nb.njit((nb.float64[:], nb.float64[:]))
def _expanding_covariance_of_returns(ret1, ret2):
    """
    Uses an expanding mean instead of a full sample mean
    to calculate covariances. May be considered a stylistic choice
    or preventing look-ahead bias, case can be made for either.
    """
    N = len(ret1)
    out = np.empty((N,))
    mean1 = ret1[0]
    mean2 = ret2[0]
    for i in range(N):
        v1 = ret1[i]
        v2 = ret2[i]
        mean1 += (v1-mean1) / (i+1)
        mean2 += (v2-mean2) / (i+1)
        out[i] = (v1-mean1) * (v2-mean2)
    return out


@nb.njit((nb.float64[:], nb.int64[:], nb.int64, nb.boolean))
def _returns(arr, timestamps, timeframe, closed_left=False):
    """
    Calculates based on a timeframe offset.
    Useful for timebased windows with variable sampling (I.e unequal time)
    The 'left' (l) side of the window is adjusted to keep within a distance
    from the current idx being filled.

    closed_left: Indicates behaviour for when a datapoint is not available
                 which is exactly distance 'timeframe' before the current idx.
                 If true:
                     Use the most recent data point within the timeframe period.
                 if false:
                     Use the most recent data point before the timeframe period.
    """
    N = len(arr)
    l = 0
    r = 0
    out = np.empty((N, ))
    if not closed_left:
        while r < N:
            while timestamps[l + 1] <= (timestamps[r] - timeframe):
                l += 1
            if timestamps[l] <= (timestamps[r] - timeframe):
                out[r] = 1 - arr[r] / arr[l]
            else:
                out[r] = np.nan
            r += 1
    elif closed_left:
        while r < N:
            while timestamps[l] < (timestamps[r] - timeframe):
                l += 1
            if timestamps[l] >= (timestamps[r] - timeframe) and l < r:
                out[r] = 1 - arr[r] / arr[l]
            else:
                out[r] = np.nan
            r += 1
    return out


def _drawdowns(arr):
    """
    Returns a sorted dataframe of the minimum drawdowns for all peaks
    where a peak is highest val up to that point (using an expanding max).
    Row 0 will contain the biggest drawdown as well the index when it occurred.
    """
    peaks = arr.expanding().max()
    df = pd.DataFrame({'peaks': peaks, 'valleys': peaks / arr})
    return df.reset_index().groupby('peaks').min().drop('peaks',
                                                        1).sort_values()
