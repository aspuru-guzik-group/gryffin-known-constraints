"""
CatDejong domain constraint for Dragonfly

def is_feasible(self, params):
    x0 = params['x0']  # str
    x1 = params['x1']  # str
    Xi = self.str2array([x0, x1])
    x0 = Xi[0]  # float
    x1 = Xi[1]  # float
    if x0 in [9, 11]:
        return False
    if x1 in [9, 11]:
        return False
    return True
"""
import numpy as  np

def str2array(sample):
    return np.array([round(float(entry[2:])) for entry in np.squeeze(sample)])

def constraint(x):
    """ Evatuates the constraint """
    x0 = x[0]
    x1 = x[1]

    Xi = str2array([x0, x1])

    x0 = Xi[0]  # float
    x1 = Xi[1]  # float
    if x0 in [9, 11]:
        return False
    if x1 in [9, 11]:
        return False
    return True
