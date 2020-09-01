import timeit
from collections import deque
from random import sample
import numpy as np
from madigan.utils import SARSD, time_profile

def test_basic_buffers():
    # _list = list(range(1000000))
    # _deque= deque(_list)
    # _np = np.arange(1000000)
    sarsd = SARSD(np.ones((1, 64, 64)), 1, 1, 1, np.ones((1, 64, 64)))

    def list_grow():
        _list = []
        for i in range(1000000):
            _list.append(sarsd)
        return _list
    def list_allocate():
        _list = [None]*1000000
        for i in range(1000000):
            _list[i] = sarsd
        return _list
    def deque_grow():
        _deque=deque(maxlen=1000000)
        for i in range(1000000):
            _deque.append(sarsd)
        return _deque
    def np_allocate():
        _np = np.empty((1000000), dtype=object)
        # _np = np.empty((1), dtype=object)
        for i in range(1000000):
            _np[i] = sarsd
        return _np

    print('\n')
    print("="*80)
    print("Testing replay buffer allocation and sampling methods \n")
    print("="*80)
    print("1. Data Structures for allocation \n")

    out = time_profile(10, 1, list_grow____=list_grow, list_allocate=list_allocate,
                       deque_grow___=deque_grow,np_allocate__=np_allocate)
    l, d, a = out['list_grow____'], out['deque_grow___'], out['np_allocate__']

    def list_sample():
        _sample = sample(l, 32)
    def deque_sample():
        _sample = sample(d, 32)
    def np_sample():
        _sample = np.random.choice(a, 32)

    print("="*80)
    print("2. Data Structures for sampling \n")
    time_profile(1000, 0, list_sample_=list_sample,
                 deque_sample=deque_sample,
                 np_sample___=np_sample)


if __name__=="__main__":
    test_basic_buffers()
