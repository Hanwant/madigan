import math
import numpy as np
import numba
from algotrading_utils.utils import time_profile
# from cpp.build.synth_c import SineGen


@numba.njit
def sine_generator(mu=0., amp=1., dx=0.1):
    x = 0.
    while True:
        yield mu + amp*math.sin(x)
        x += dx


def test_gen_speed(gen):
    for i in range(10000):
        res = next(gen)
    return res


if __name__ == "__main__":
    import timeit
    mu = 0.
    amp = 1.0
    dx = 0.01
    gen1 = iter(sine_generator(mu, amp, dx))
    time_profile(100, 1, gen1=lambda: test_gen_speed(gen1))

    # total_time = timeit.timeit("test_gen_speed(env1)",
    #                            setup="""from synth import Synth, test_gen_speed; env1 = Synth() """,
    #                            number=100)
    # print("Time taken as per timeit: ", total_time/100)
