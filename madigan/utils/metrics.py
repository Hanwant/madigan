import itertools
from typing import Union, Iterable

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


def make_timeframes(timestamps: Iterable, is_datetime: bool) -> list:
    """
    Based on timestamps range, makes timeframe offsets for use in
    returns metrics such as sharpe or beta coefficient.
    If timestamps are not datetime, a list of integer offsets will be returned.
    The offsets are constructed by taking powers of 2 (starting from 0)
    up to len(timestamps) / 10 so that there are at least 10 samples for the
    highest timeframe.
    If is_datetime, a list of pd.TimeDelta objects are returned using a fixed
    list of timeframes. Timeframes are returned if timestamps contains at least
    10 samples for that timeframe:
        timeframes = filter(timeframe <= (timestamps[-1] - timestamps[0])/10)

    returns: a list of tuples where the first pos in the tuple is a
             human readable name for the timeframe and second pos is
             the tf
    """
    dt_tfs = [(tf, pd.Timedelta(tf)) for tf in ('1min', '10min', '30min', '1h',
                                                '2h', '4h', '8h', '1d')]
    if not is_datetime:
        max_tf = int(np.log2(len(timestamps) // 10))
        return [(str(2**i), 2**i) for i in range(max_tf)]
    max_tf = pd.to_datetime(timestamps[-1]) - pd.to_datetime(timestamps[0])
    return list(filter(lambda x: x[1] * 10 <= max_tf, dt_tfs))


def test_summary(test_metrics: Union[dict, pd.DataFrame],
                 timeframes: list,
                 assets: list = None,
                 is_datetime: bool = False) -> pd.DataFrame:
    """
    Tearsheet
    Given metrics from a test episode, returns a tearsheet-like summary.
    Needs to be updated to include backtest style stats
    I.e drawdown, sharpe etc - needs timestamps
    """
    # HOUSEKEEPING ##########################################################
    df = pd.DataFrame(test_metrics)
    df.set_index('timestamp', inplace=True)
    prices = np.array(test_metrics['prices'].tolist())
    n_assets = prices.shape[1]
    assets = assets if assets is not None else [
        f'asset_{i}' for i in range(n_assets)
    ]
    if n_assets != len(assets):
        raise ValueError(
            "prices data (dim 1) and assets list passed are not of same length"
        )
    if is_datetime:
        timestamps = pd.to_datetime(df.index).values.astype(np.int64)
    else:
        timestamps = df.index.values

    max_tf = (timestamps[-1] - timestamps[0]) // 10

    # BETA COEFFICIENTS  ######################################################
    betas = {}
    for (i, asset), (tf_name, tf) in itertools.product(enumerate(assets),
                                                       timeframes):
        if tf <= max_tf:
            betas[f'beta_{asset}_offset_{tf_name}'] = beta_coeff_of_series(
                test_metrics['equity'], prices[:, i], tf, timestamps)
        else:
            betas[f'beta_{asset}_offset_{tf_name}'] = np.nan

    # TIME SPENT IN POSITIONS
    time_spent_in_pos = {}
    positions = np.array(df['ledger'].tolist())
    for i, asset in enumerate(assets):
        time_spent = len(np.where(positions[:, i] != 0.)[0])
        time_spent = time_spent / len(positions)  # as proportion
        time_spent_in_pos[f'time_spent_in_pos_{asset}'] = time_spent

    # RETURNS + VOL ADJUSTED RETURNS  #########################################
    returns = {}
    equity = np.array(df['equity'], copy=False)
    for tf_name, tf in timeframes:
        if tf <= max_tf:
            log_returns = _returns(equity,
                                   timestamps,
                                   tf,
                                   closed_left=False,
                                   log_returns=True)
            returns[f'equity_returns_offset_{tf_name}'] = np.nanmean(
                log_returns)
            returns[f'equity_sharpe_offset_{tf_name}'] = _sharpe_of_returns(
                log_returns)
            returns[f'equity_sortino_offset_{tf_name}'] = _sortino_of_returns(
                log_returns)
        else:
            returns[f'equity_returns_offset_{tf_name}'] = np.nan
            returns[f'equity_sharpe_offset_{tf_name}'] = np.nan
            returns[f'equity_sortino_offset_{tf_name}'] = np.nan

    # MODEL OUTPUTS OF INTEREST  #############################################
    qvals = {}
    for qval in ('qvals', 'qvals1', 'qvals2'):  # 1 and 2 for twin Q Netowrks
        if qval in df.columns:
            qvals['mean_' + qval] = np.array(df[qval].tolist()).mean()

    out = {
        'mean_equity': df['equity'].mean(),
        'final_equity': df['equity'].iloc[-1],
        'mean_reward': df['reward'].mean(),
        'max_drawdown': _drawdowns(df['equity']).iloc[0]['valleys'],
        'mean_transaction_cost':
        np.array(df['transaction_cost'].tolist()).mean(),
        'total_transaction_cost':
        np.array(df['transaction_cost'].tolist()).sum(),
        'nsteps': len(df),
        **betas,
        **time_spent_in_pos,
        **returns,
        **qvals
    }
    return pd.DataFrame({k: [v] for k, v in out.items()})


def variance(timeseries: Union[np.ndarray, pd.Series],
             timeframe: Union[int, pd.Timedelta],
             timestamps: np.ndarray = None,
             ddof: int = 1,
             log_returns: bool = True):
    """
    Computes the variance of returns of a given timeseries
    I.e price or equity curve.
    @params
    timeseries: array = the timeseries to compute returns variance for
    timeframe: window = the timeframe for return calculation.
                        (n_periods if timestamps are integer increments)
    ddof: int = degrees of freedom in vairance calculation. Default = 1.
    log_returns: bool = whether to use log or normal returns default=True
    """
    if timestamps is None:
        if isinstance(timeseries, pd.Series):
            if isinstance(timeseries.index, pd.DatetimeIndex):
                timestamps = timeseries.index.values.view(np.int64)
            else:
                timestamps = timeseries.index.values
        else:
            timestamps = np.arange(len(timeseries))

    arr = np.array(timeseries)
    if isinstance(timeframe, pd.Timedelta):
        timeframe = timeframe.view(np.int64)
    returns = _returns(arr, timestamps, timeframe, log_returns=log_returns)
    return np.nanvar(returns, ddof=ddof)


def covariance_of_series(series_a: Union[np.ndarray, pd.Series],
                         series_b: Union[np.ndarray, pd.Series],
                         timeframe: Union[int, pd.Timedelta],
                         timestamps: np.ndarray = None,
                         ddof: int = 1,
                         log_returns: bool = True):
    """
    Computes the covariance of returns for 2 timeseries.
    I.e price vs equity curve.
    @params
    series_a: array
    series_b: array
    timeframe: window = the timeframe for return calculation.
                        (n_periods if timestamps are integer increments)
    ddof: int = degrees of freedom in vairance calculation. Default = 1.
    log_returns: bool = uses log returns if true, else normal returns
    """
    if not (isinstance(series_a, (np.ndarray, pd.Series))
            and isinstance(series_b, (np.ndarray, pd.Series))):
        raise ValueError(
            "Inputs to covariance of series must be either np.ndarray or pd.Series"
        )
    if timestamps is None:
        if isinstance(series_a, pd.Series):
            if isinstance(series_a.index, pd.DatetimeIndex):
                timestamps = series_a.index.values.view(np.int64)
            else:
                timestamps = series_a.index.values
        else:
            timestamps = np.arange(len(series_a))

    arr_1 = np.array(series_a)
    arr_2 = np.array(series_b)
    if isinstance(timeframe, pd.Timedelta):
        timeframe = timeframe.view(np.int64)
    ret_1 = _returns(arr_1,
                     timestamps,
                     timeframe,
                     closed_left=False,
                     log_returns=log_returns)
    ret_2 = _returns(arr_2,
                     timestamps,
                     timeframe,
                     closed_left=False,
                     log_returns=log_returns)
    cov = _covariance_of_returns(ret_1, ret_2, ddof=1)
    # cov = _expanding_covariance_of_returns(returns_1, returns_2)
    return cov


def beta_coeff_of_series(series_a,
                         series_b,
                         timeframe,
                         timestamps=None,
                         log_returns=True):
    """Treats series_a as strategy returns and series_b as the market (price)
    """
    if not (isinstance(series_a, (np.ndarray, pd.Series))
            and isinstance(series_b, (np.ndarray, pd.Series))):
        raise ValueError(
            "Inputs to beta coeff must be either np.ndarray or pd.Series")

    if timestamps is None:
        if isinstance(series_a, pd.Series):
            if isinstance(series_a.index, pd.DatetimeIndex):
                timestamps = series_a.index.values.view(np.int64)
            else:
                timestamps = series_a.index.values
        else:
            timestamps = np.arange(len(series_a))

    arr_1 = np.array(series_a, copy=False)
    arr_2 = np.array(series_b, copy=False)
    if isinstance(timeframe, pd.Timedelta):
        timeframe = timeframe.view(np.int64)
    ret_1 = _returns(arr_1,
                     timestamps,
                     timeframe,
                     closed_left=False,
                     log_returns=log_returns)
    ret_2 = _returns(arr_2,
                     timestamps,
                     timeframe,
                     closed_left=False,
                     log_returns=log_returns)
    beta = _beta_coeff_of_returns(ret_1, ret_2, ddof=1)
    return beta


def _covariance_of_returns(ret_1: np.ndarray, ret_2: np.ndarray, ddof=1):
    """ Computes standard covariance of two returns arrays """
    cov = (ret_1 - np.nanmean(ret_1)) * (ret_2 - np.nanmean(ret_2))
    return np.nansum(cov) / (len(cov) - ddof)


@nb.njit((nb.float64[:], nb.float64[:], nb.int64))
def _expanding_covariance_of_returns(ret1, ret2, ddof=1):
    """
    Uses an expanding mean instead of a full sample mean
    to calculate covariances. May be considered a stylistic choice
    or preventing look-ahead bias, case can be made for either.
    """
    N = len(ret1)
    out = np.empty((N, ))
    mean1 = ret1[0]
    mean2 = ret2[0]
    for i in range(N):
        v1 = ret1[i]
        v2 = ret2[i]
        if v1 != np.nan:
            mean1 += (v1 - mean1) / i
        if v2 != np.nan:
            mean2 += (v2 - mean2) / i
        if (v1 != np.nan) and (v2 != np.nan):
            out[i] = ((v1 - mean1) * (v2 - mean2))
        else:
            out[i] = np.nan
    return np.nansum(out) / (len(out) - ddof)


def _beta_coeff_of_returns(ret_1, ret_2, ddof=1):
    """ Treats ret_2 as market returns with which to normalize"""
    cov = _covariance_of_returns(ret_1, ret_2, ddof=ddof)
    var = np.nanvar(ret_2, ddof=ddof)
    return cov / var


def _sharpe_of_returns(returns, benchmark=None, ddof=1):
    if benchmark is not None:
        if not isinstance(benchmark, float):
            if not isinstance(benchmark, np.ndarray):
                raise ValueError("benchmark should be ndarray")
    else:
        benchmark = 0.
    diff = returns - benchmark
    diff_mean = np.nanmean(diff)
    diff_std = np.nanstd(diff, ddof=ddof)
    if diff_std == 0.:
        return 0.
    return diff_mean / diff_std


def _sortino_of_returns(returns, benchmark=None, ddof=1):
    if benchmark is not None:
        if not isinstance(benchmark, float):
            if not isinstance(benchmark, np.ndarray):
                raise ValueError("benchmark should ndarray")
    else:
        benchmark = 0.
    diff = returns - benchmark
    downside = (diff[np.where(diff < 0.)[0]]**2).sum() / (len(diff) - ddof)
    diff_mean = np.nanmean(diff)
    if downside == 0.:
        if diff_mean > 0.:
            return 10.  # heuristic for no downside
        else:
            return 0.  # heuristic for no down or upside
    return np.clip(diff / downside, None, 10.)


@nb.njit((nb.float64[:], nb.int64[:], nb.int64, nb.boolean, nb.boolean))
def _returns(arr, timestamps, timeframe, closed_left=False, log_returns=True):
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
                if log_returns:
                    out[r] = np.log(arr[r] / arr[l])
                else:
                    out[r] = 1 - arr[r] / arr[l]
            else:
                out[r] = np.nan
            r += 1
    elif closed_left:
        while r < N:
            while timestamps[l] < (timestamps[r] - timeframe):
                l += 1
            if timestamps[l] >= (timestamps[r] - timeframe) and l < r:
                if log_returns:
                    out[r] = np.log(arr[r] / arr[l])
                else:
                    out[r] = 1 - arr[r] / arr[l]
            else:
                out[r] = np.nan
            r += 1
    return out


def _drawdowns(arr: pd.Series):
    """
    Returns a sorted dataframe of the max drawdowns for all peaks, where peaks
    are determined from a walk-forward perspective (using expanding max).
    Row 0 will contain the biggest drawdown as well the index when it occurred.
    """
    peaks = arr.expanding().max()
    df = pd.DataFrame({'peaks': peaks, 'valleys': arr / peaks})
    return df.reset_index().groupby('peaks').min().sort_values(by='valleys')
