import timeit
from collections import deque
from random import sample
import numpy as np
from madigan.utils import SARSD, State, time_profile
from madigan.utils import ReplayBuffer
import torch


def test_data_structures():
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
        for i in range(1000000):
            _np[i] = sarsd
        return _np
    # def blist_grow():
    #     _list = blist()
    #     for i in range(1000000):
    #         _list.append(sarsd)
    #     return _list
    # def blist_allocate():
    #     _list = blist([None])*1000000
    #     for i in range(1000000):
    #         _list[i] = sarsd
    #     return _list

    print('\n')
    print("="*80)
    print("Testing data structures for allocation and sampling methods \n")
    print("="*80)
    print("1. Data Structures for allocation \n")

    out = time_profile(10, 1, list_grow____=list_grow, list_allocate=list_allocate,
                       deque_grow___=deque_grow, np_allocate__=np_allocate,)
                       # blist_grow___=blist_grow, blist_allocate=blist_allocate)
    l, d, a = out['list_grow____'], out['deque_grow___'], out['np_allocate__']#, out['blist_grow___']

    def list_sample():
        _sample = sample(l, 32)
    def deque_sample():
        _sample = sample(d, 32)
    def np_sample():
        _sample = np.random.choice(a, 32)
    # def blist_sample():
    #     _sample = sample(b, 32)

    print("="*80)
    print("2. Data Structures for sampling \n")
    time_profile(1000, 0, list_sample_=list_sample,
                 deque_sample=deque_sample,
                 np_sample___=np_sample,)
                 # blist_sample=blist_sample)
    print("="*80)

def test_replay_buffer():
    size = 100_000
    rb = ReplayBuffer(size)
    state = State(np.random.randn(12, 64), np.random.randn(12), 0)
    sarsd = SARSD(state, 1, 2., state, False)

    def sample_2(rb, n):
        idx = sample(range(rb.filled), n)
        state_price = np.array([rb._buffer[i].state.price for i in idx])
        state_portfolio = np.array([rb._buffer[i].state.portfolio for i in idx])
        state_timestamp= np.array([rb._buffer[i].state.timestamp for i in idx])
        state = State(state_price, state_portfolio, state_timestamp)
        reward = np.array([rb._buffer[i].reward for i in idx])
        action = np.array([rb._buffer[i].action for i in idx])
        next_state_price = np.array([rb._buffer[i].next_state.price for i in idx])
        next_state_portfolio = np.array([rb._buffer[i].next_state.portfolio for i in idx])
        next_state_timestamp = np.array([rb._buffer[i].next_state.timestamp for i in idx])
        next_state = State(next_state_price, next_state_portfolio, next_state_timestamp)
        done = np.array([rb._buffer[i].done for i in idx])
        return SARSD(state, action, reward, next_state, done)

    def sample_3(rb, n):
        idx = sample(range(rb.filled), n)
        state_price = torch.tensor([rb._buffer[i].state.price for i in idx])
        state_portfolio = torch.tensor([rb._buffer[i].state.portfolio for i in idx])
        state_timestamp = torch.tensor([rb._buffer[i].state.timestamp for i in idx])
        state = State(state_price, state_portfolio, state_timestamp)
        reward = torch.tensor([rb._buffer[i].reward for i in idx])
        action = torch.tensor([rb._buffer[i].action for i in idx])
        next_state_price = torch.tensor([rb._buffer[i].next_state.price for i in idx])
        next_state_portfolio = torch.tensor([rb._buffer[i].next_state.portfolio for i in idx])
        next_state_timestamp = torch.tensor([rb._buffer[i].next_state.timestamp for i in idx])
        next_state = State(next_state_price, next_state_portfolio, next_state_timestamp)
        done = torch.tensor([rb._buffer[i].done for i in idx])
        return SARSD(state, action, reward, next_state, done)

    def sample_4(rb, n):
        sarsd = rb.sample(n)
        state = State(torch.tensor(sarsd.state.price), torch.tensor(sarsd.state.portfolio), 
                torch.tensor(sarsd.state.timestamp))
        action = torch.tensor(sarsd.action)
        next_state = State(torch.tensor(sarsd.next_state.price), torch.tensor(sarsd.next_state.portfolio), 
                torch.tensor(sarsd.next_state.timestamp))
        reward = torch.tensor(sarsd.reward)
        done = torch.tensor(sarsd.done)
        return SARSD(state, action, reward, next_state, done)

    def sample_5(rb, n):
        sarsd = sample_2(rb, n)
        state = State(torch.tensor(sarsd.state.price), torch.tensor(sarsd.state.portfolio),
                torch.tensor(sarsd.state.timestamp))
        action = torch.tensor(sarsd.action)
        next_state = State(torch.tensor(sarsd.next_state.price), torch.tensor(sarsd.next_state.portfolio), 
                torch.tensor(sarsd.next_state.timestamp))
        reward = torch.tensor(sarsd.reward)
        done = torch.tensor(sarsd.done)
        return SARSD(state, action, reward, next_state, done)

    def sample_6(rb, n):
        _sample = sample(rb._buffer, n)
        state_price = np.stack([s.state.price for s in _sample])
        state_portfolio = np.stack([s.state.portfolio for s in _sample])
        state_timestamp = np.stack([s.state.timestamp for s in _sample])
        state = State(state_price, state_portfolio, state_timestamp)
        next_state_price = np.stack([s.next_state.price for s in _sample])
        next_state_portfolio = np.stack([s.next_state.portfolio for s in _sample])
        next_state_timestamp = np.stack([s.next_state.timestamp for s in _sample])
        next_state = State(next_state_price, next_state_portfolio, next_state_timestamp)
        action = np.array([s.action for s in _sample])
        reward = np.array([s.reward for s in _sample])
        done = np.array([s.done for s in _sample])
        return SARSD(state, action, reward, next_state, done)

    def populate_buffer():
        for i in range(size):
            rb.add(sarsd)
    def sample_buffer():
        rb.sample(32)
    def sample_buffer_2():
        sample_2(rb, 32)
    def sample_buffer_3():
        sample_3(rb, 32)
    def sample_buffer_4():
        sample_4(rb, 32)
    def sample_buffer_5():
        sample_5(rb, 32)
    def sample_buffer_6():
        sample_6(rb, 32)

    print('\n')
    print("="*80)
    print("Testing implemented replay buffer allocation and sampling methods \n")
    print("="*80)
    print("1. Allocation \n")
    time_profile(10, 0, populate_buffer=populate_buffer)
    print("="*80)
    print("2. Sampling \n")
    time_profile(1000, 0, sample_buffer_default____=sample_buffer, sample_buffer_def_v2_____=sample_buffer_2,
                 sample_buffer_torch______=sample_buffer_3, sample_buffer_np_to_torch=sample_buffer_4,
                 sample_buffer_np_torch_v2=sample_buffer_5, sample_buffer_np_stack___=sample_buffer_6)
    print("="*80)


if __name__=="__main__":
    test_data_structures()
    test_replay_buffer()
