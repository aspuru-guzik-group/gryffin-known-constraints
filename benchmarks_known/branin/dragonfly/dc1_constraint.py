"""
Branin domain constraint for Dragonfly

def is_feasible(self, params):
    x0 = params['x0']
    x1 = params['x1']

    y0 = (x0-0.12389382)**2 + (x1-0.81833333)**2
    y1 = (x0-0.961652)**2 + (x1-0.165)**2

    if y0 < 0.2**2 or y1 < 0.35**2:
        return False
    else:
        return True
"""

def constraint(x):
    """ Evatuates the constraint """
    x0 = x[0]
    x1 = x[1]

    y0 = (x0-0.12389382)**2 + (x1-0.81833333)**2
    y1 = (x0-0.961652)**2 + (x1-0.165)**2

    if y0 < 0.2**2 or y1 < 0.35**2:
        return False
    else:
        return True
