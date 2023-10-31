from enum import Enum
from timeit import default_timer as timer

MaxDepth = 20
Infinity = 10000
Win = 1000
Loss = -Win
NoMove = (0, 0, -1)


class AbortType(Enum):
    ITERATIONS = 1,
    NODES = 2,
    TIME = 3


class Aborter:

    def __init__(self, a_type: AbortType, value: int):
        self._abort_type = a_type
        self._value = value
        self.reset()

    def reset(self):
        if self._abort_type == AbortType.TIME:
            self._n = timer() + (self._value / 1000.0)
        else:
            self._n = self._value

    def do_abort(self, depth: int, nodes: int):
        if self._value == 0:
            return False
        if self._abort_type == AbortType.TIME:
            return timer() >= self._n
        elif self._abort_type == AbortType.NODES:
            return nodes > self._n
        else:
            return depth > self._n


class Avg:

    def __init__(self):
        self.n = 0
        self.avg = 0.0

    def __repr__(self):
        return str(round(self.avg,2)) + f'({self.n})'

    def add(self, v):
        self.n += 1
        self.avg += (v - self.avg) / self.n


def argmax(data, n, evaluate, maximum_value=None):
    max_i = 0
    max_v = evaluate(data, 0)
    if max_v == maximum_value:
        return max_i
    for i in range(n):
        value = evaluate(data, i)
        if value == maximum_value:
            return i
        if value > max_v:
            max_i = i
            max_v = value
    return max_i
