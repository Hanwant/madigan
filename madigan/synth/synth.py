import math
import numpy as np
import numba
from algotrading_utils.utils import time_profile
# from cpp.build.synth_c import SineGen


@numba.njit
def single_sine_gen(freq=1., mu=0., amp=1., dx=0.1):
    x = 0.
    while True:
        yield mu + amp*math.sin(2*math.pi*x*freq)
        x += dx


# Slightly slower than numpy.numba version below
# def multi_sine_gen(*single_gens):
#     while True:
#         yield tuple((next(gen) for gen in single_gens))

@numba.njit
def multi_sine_gen(_params, dx=0.01):
    """
    Returns 1d np array of a multi-dim sine wave process
    Usage: Pass in a numpy array containing rows of parameters (3 cols each)
    for sine waves to be generated.
    The columns comprise 4 parameters for each sine wave:
        freq:  frequency multiple of 2pi
        mu:    offset (average y)
        amp:   amplitude multipler
        phase: initial offset
        I.e
        params = np.array([[1., 0., 1., 1.],   # Sine wave with freq=1., mu=0., amp=1., phase=1.
                           [2., 1., 2., np.pi] # freq=2., mu=1., amp=2., phase=pi
                           ])
    """
    # params = np.array(param_list)
    params = _params.copy()
    while True:
        yield params[:, 1] + params[:, 2] * np.sin(2 * np.pi * params[:, 0] * params[:, 3])
        params[:, 3] += dx

def test_gen_speed(gen):
    for i in range(10000):
        res = next(gen)
    return res


if __name__ == "__main__":
    import timeit
    import matplotlib.pyplot as plt
    freq=1.
    mu = 0.
    amp = 1.0
    dx = 0.01
    gen1 = iter(single_sine_gen(freq, mu, amp,dx))
    time_profile(100, 1, gen1=lambda: test_gen_speed(gen1))

    gen1 = iter(single_sine_gen(freq, mu, amp, dx))
    singles = []
    for i in range(1000):
        dat = next(gen1)
        singles.append(dat)
    if True:
        freq = [1., 2., 3., 4.,]
        mu = [2., 3, 4., 5.] # Keeps negative prices from ocurring
        amp = [1., 2., 3., 4.]
        phase = [0., 1., 2., 0.]
        params = np.stack([freq, mu, amp, phase], axis=1)
        multi_gen = multi_sine_gen(params, dx=0.01)
        multis = []
        for i in range(1000):
            dat = next(multi_gen)
            multis.append(dat)


    fig, ax = plt.subplots(1, 2)
    ax[0].plot(singles)
    ax[1].plot(np.array(multis))
    plt.show()
    # total_time = timeit.timeit("test_gen_speed(env1)",
    #                            setup="""from synth import Synth, test_gen_speed; env1 = Synth() """,
    #                            number=100)
    # print("Time taken as per timeit: ", total_time/100)
