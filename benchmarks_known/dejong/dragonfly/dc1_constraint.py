"""
Dejong domain constraint for Dragonfly

def is_feasible(self, params):
    x0 = params['x0']
    x1 = params['x1']
    y = (x0-0.5)**2 + (x1-0.5)**2

    if np.abs(x0-x1) < 0.1:
        return False

    if 0.05 < y < 0.15:
        return False
    else:
        return True
"""
import numpy as np

def constraint(x):
    """ Evatuates the constraint """
    x0 = x[0]
    x1 = x[1]
    y = (x0-0.5)**2 + (x1-0.5)**2

    if np.abs(x0-x1) < 0.1:
        return False

    if 0.05 < y < 0.15:
        return False
    else:
        return True
