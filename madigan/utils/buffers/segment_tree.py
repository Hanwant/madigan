"""
For use in prioritized replay buffer
"""
import operator


class SegmentTree:
    def __init__(self, size, op, default_init):
        self.size = size
        self.op = op
        self.default_init = default_init
        self._values = [default_init for _ in range(2 * size)]

    def reduce(self, start, end):
        """ Iterative version """
        start += self.size
        end += self.size
        res = self.default_init
        while start < end:
            if (start % 2) == 1:
                res = self.op(res, self._values[start])
                start += 1
            if (end % 2) == 1:
                end -= 1
                res = self.op(res, self._values[end])
            start //= 2
            end //= 2
        return res

    def __setitem__(self, idx, val):
        if not (0 <= idx < self.size):
            raise IndexError("idx to segment tree must be 0 < idx < tree.size")
        idx += self.size
        self._values[idx] = val

        idx //= 2
        while idx >= 1:
            self._values[idx] = self.op(self._values[2 * idx],
                                        self._values[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        if not (0 <= idx < self.size):
            raise IndexError("idx to segment tree must be 0 < idx < tree.size")
        return self._values[idx + self.size]


class SumTree(SegmentTree):
    def __init__(self, size):
        super().__init__(size, operator.add, 0.)

    def find_prefixsum_idx(self, prefixsum: float):
        idx = 1
        while idx < self.size:
            left = 2 * idx
            if self._values[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self._values[left]
                idx = left + 1
        return idx - self.size


class MinTree(SegmentTree):
    def __init__(self, size):
        super().__init__(size, min, float("inf"))
