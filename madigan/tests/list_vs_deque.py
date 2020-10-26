from timeit import timeit
from collections import deque
import numpy as np

lst = [float(i) for i in range(10000)]
deq = deque(lst)
lst_smalls = [[float(i) for i in range(10)] for j in range(1000)]
deq_smalls = [deque(lst_smalls[j]) for j in range(1000)]


def test_array_creation_speed():
    lst_time =  timeit(lambda: np.array(lst), number=1000)
    deq_time =  timeit(lambda: np.array(deq), number=1000)
    print('time taken to make ndarray from list: ', lst_time)
    print('time taken to make ndarray from deque: ', deq_time)

def test_popping():
    lst_time =  timeit(lambda: lst.pop(0) ,number=1000)
    deq_time =  timeit(lambda: deq.popleft(), number=1000)
    print(f'time taken to pop from list of size {len(lst)}: ', lst_time)
    print(f'time taken to pop from deque of size {len(deq)}: ', deq_time)
    lst_times = [timeit(lambda: lst_smalls[i].pop(0) ,number=10) for i in range(1000) ]
    deq_times = [timeit(lambda: deq_smalls[i].popleft() ,number=10) for i in range(1000) ]
    print(f'time taken to pop from list of size 10', sum(lst_times)/len(lst_times))
    print(f'time taken to pop from deque of size 10', sum(deq_times)/len(deq_times))


if __name__=="__main__":
    test_array_creation_speed()
    test_popping()
