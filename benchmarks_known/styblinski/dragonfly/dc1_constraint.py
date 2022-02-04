"""
StyblinskiTang domain constraint for Dragonfly

def is_feasible(self, params):
    x0 = params['x0']
    x1 = params['x1']

    if x0+x1 < 0.4:
        return False
    if x0 > 0.6 and x1 > 0.6:
        return False
    if x0 < 0.4 and x1 > 0.6:
        return False
    if x0 > 0.6 and x1 < 0.4:
        return False
    return True
"""

def constraint(x):
    """ Evatuates the constraint """
    x0 = x[0]
    x1 = x[1]

    if x0+x1 < 0.4:
        return False
    if x0 > 0.6 and x1 > 0.6:
        return False
    if x0 < 0.4 and x1 > 0.6:
        return False
    if x0 > 0.6 and x1 < 0.4:
        return False
    return True
