"""
Schwefel domain constraint for Dragonfly

def is_feasible(self, params):
    x0 = params['x0']
    x1 = params['x1']
    Xi = np.array([x0, x1])
    for c, r in zip(self.centers, self.radii):
        if np.linalg.norm(c - Xi) < r:
            return False

    return True
"""

import numpy as np

np.random.seed(42)
N = 20
centers = [np.random.uniform(low=0.0, high=1.0, size=2) for i in range(N)]
radii = [np.random.uniform(low=0.05, high=0.15, size=1) for i in range(N)]

def constraint(x):
    """ evaluates the constraint """
    x0 = x[0]
    x1 = x[1]
    Xi = np.array([x0, x1])
    for c, r in zip(centers, radii):
        if np.linalg.norm(c - Xi) < r:
            return False

    return True
