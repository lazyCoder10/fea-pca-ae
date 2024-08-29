import numba
from numba import njit
import numpy as np
from numpy import cos, sqrt, pi, e, exp

"""
Continuous benchmark functions wrappers to speed up calculations
"""

@njit
def sphere__(solution=None):
    return np.sum(solution ** 2)


@njit
def elliptic__(solution=None):
    result = 0.0
    for i in range(0, len(solution)):
        result += (10 ** 6) ** (i / (len(solution) - 1)) * solution[i] ** 2
    return result


@njit
def rastrigin__(solution=None):
    return np.sum(solution ** 2 - 10 * cos(2 * pi * solution) + 10)


@njit
def ackley__(solution=None):
    return -20 * exp(-0.2 * sqrt(np.sum(solution ** 2) / len(solution))) - exp(
        np.sum(cos(2 * pi * solution)) / len(solution)) + 20 + e


#@njit
def schwefel__(solution=None):
    return np.sum([np.sum(solution[:i]) ** 2 for i in range(0, len(solution))])


@njit
def rosenbrock__(solution=None):
    result = 0.0
    for i in range(len(solution) - 1):
        result += 100 * (solution[i] ** 2 - solution[i + 1]) ** 2 + (solution[i] - 1) ** 2
    return result
