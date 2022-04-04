"""
NarrowFunnel domain constraint for Dragonfly

x0 = params['x0']
x1 = params['x1']
is_feas = True

if np.logical_or(
    0.41 < x0 < 0.46,
    0.54 < x0 < 0.59,
):
    is_feas = False

if np.logical_or(
    0.34 < x1 < 0.41,
    0.59 < x1 < 0.66,
):
    is_feas = False

return is_feas

"""

import numpy as np


def constraint(x):
    """ evaluates the constraint """
    x0 = x[0]
    x1 = x[1]
    is_feas = True

    if np.logical_or(
        0.41 < x0 < 0.46,
        0.54 < x0 < 0.59,
    ):
        is_feas = False

    if np.logical_or(
        0.34 < x1 < 0.41,
        0.59 < x1 < 0.66,
    ):
        is_feas = False

    return is_feas
