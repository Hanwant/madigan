import argparse

import numpy as np
import pandas as pd
import numba

from madigan.environments.reward_shaping import SharpeFixedWindow, SharpeEWMA

@numba.jit((numba.float64[:], numba.int64), nopython=True)
def ewstd_np(arr, window):
    """
    exp weighted moving std - pd.ewm(span=window).std()
    Template for SharpeEWMA
    """
    n = arr.shape[0]
    ewma = np.empty(n, dtype=np.float64)
    ewstd = np.empty(n, dtype=np.float64)
    alpha = 2 / float(window + 1)
    ewma_old = arr[0]
    ewstd_old = 0.
    ewma[0] = ewma_old
    ewstd[0] = ewstd_old
    w = 1.
    w2 = 1.
    for i in range(1, n):
        w += (1 - alpha) ** i
        w2 += ((1 - alpha) ** i) ** 2
        ewma_old = ewma_old * (1 - alpha) + arr[i]
        ewma[i] = ewma_old / w
        ewstd_old = ewstd_old * (1 - alpha) + ((arr[i] - ewma[i]) * (arr[i] - ewma[i - 1]))
        ewstd[i] = ewstd_old / (w - w2 / w)
    ewstd = np.sqrt(ewstd)
    ewstd[0] = np.nan
    return ewstd


def test_sharpe_fixed_window():
    rewards = np.random.randn(1000)
    window = 50
    shaper = SharpeFixedWindow(window)
    rolling_sharpe_ref = (rewards / pd.Series(rewards).rolling(
        window, min_periods=2).std(ddof=0))[:].values
    rolling_sharpe_ref = np.nan_to_num(rolling_sharpe_ref, 0.)
    rolling_sharpe = []
    for reward in rewards:
        rolling_sharpe.append(shaper.stream(reward))
    rolling_sharpe = np.array(rolling_sharpe[:])
    np.testing.assert_allclose(rolling_sharpe, rolling_sharpe_ref)


def test_numba_ewstd():
    arr = np.random.randn(1000)
    window = 50
    sharpe_ref = (arr / pd.Series(arr).ewm(span=window).std())[:].values
    sharpe = arr / ewstd_np(arr, window)
    np.testing.assert_allclose(sharpe, sharpe_ref)


def test_sharpe_ewma_window():
    rewards = np.random.randn(1000)
    alpha = 0.9
    window = 50
    shaper = SharpeEWMA(window)
    rolling_sharpe_ref = (rewards / pd.Series(rewards).ewm(span=window
                                                           ).std())[:].values
    rolling_sharpe_ref = np.nan_to_num(rolling_sharpe_ref, 0.)
    rolling_sharpe = []
    for reward in rewards:
        rolling_sharpe.append(shaper.stream(reward))
    rolling_sharpe = np.array(rolling_sharpe[:])
    PLOT = false
    if PLOT:
        import matplotlib.pyplot as plt
        # plt.plot(rewards, label='rewards')
        plt.plot(rolling_sharpe, label='shaper')
        plt.plot(rolling_sharpe_ref, label='ref')
        plt.legend()
        plt.show()
    np.testing.assert_allclose(rolling_sharpe, rolling_sharpe_ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    debug = args.debug

    tests = [test_sharpe_fixed_window, test_numba_ewstd,
             test_sharpe_ewma_window]
    num_tests = len(tests)
    completed = 0
    failed = []
    for i, test in enumerate(tests):
        try:
            test()
            completed += 1
        except Exception as E:
            if debug:
                raise E
            else:
                failed.append(i)

    if completed == len(tests):
        print('PASSED')
        print(f'All {completed}/{len(tests)} tests completed')
    else:
        print('FAILED')
        print(f'{completed}/{len(tests)} tests completed')
        print(f'tests which failed: {[tests[i].__name__ for i in failed]}')

