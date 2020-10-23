from timeit import timeit
from collections import deque
import numpy as np

lst = [float(i) for i in range(10000)]
deq = deque(lst)


def test_array_creation_speed():
    lst_time =  timeit(lambda: np.array(lst), number=1000)
    deq_time =  timeit(lambda: np.array(deq), number=1000)
    print('time taken to make ndarray from list: ', lst_time)
    print('time taken to make ndarray from deque: ', deq_time)

def test_popping():
    lst_time =  timeit(lambda: lst.pop(0) ,number=1000)
    deq_time =  timeit(lambda: deq.popleft(), number=1000)
    print('time taken to pop from list: ', lst_time)
    print('time taken to pop from deque: ', deq_time)


if __name__=="__main__":
    test_array_creation_speed()
    test_popping()
