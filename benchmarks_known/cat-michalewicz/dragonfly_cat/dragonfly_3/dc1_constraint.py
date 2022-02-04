"""
CatMichalewicz domain constraint for Dragonfly

def is_feasible(self, params):
    x0 = params['x0']  # str
    x1 = params['x1']  # str
    Xi = self.str2array([x0, x1])
    x0 = Xi[0]  # float
    x1 = Xi[1]  # float

    y = ((x0-14))**2 + (x1-10)**2
    if 5 < y < 30:
        return False
    if 12.5 < x0 < 15.5:
        if x1 < 5.5:
            return False
    if 8.5 < x1 < 11.5:
        if x0 < 9.5:
            return False
    return True


"""
import numpy as  np

def str2array(sample):
    return np.array([round(float(entry[2:])) for entry in np.squeeze(sample)])

def constraint(x):
    """ Evatuates the constraint """
    x0 = x[0]  # str
    x1 = x[1]  # str
    Xi = str2array([x0, x1])
    x0 = Xi[0]  # float
    x1 = Xi[1]  # float

    y = ((x0-14))**2 + (x1-10)**2
    if 5 < y < 30:
        return False
    if 12.5 < x0 < 15.5:
        if x1 < 5.5:
            return False
    if 8.5 < x1 < 11.5:
        if x0 < 9.5:
            return False
    return True
