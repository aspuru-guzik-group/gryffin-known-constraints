"""
CatCamel domain constraint for Dragonfly

def is_feasible(self, params):
    x0 = params['x0']  # str
    x1 = params['x1']  # str
    sample_tuple = (x0, x1)
    if sample_tuple in self.infeas_tuples:
        return False
    return True
"""
import numpy as  np


# choose infeasible points at random
num_dims = 2
num_opts = 21
#options = [f'x_{i}' for i in range(0, num_opts, 1)]
options = [i for i in range(0, num_opts, 1)]
num_infeas = 100
np.random.seed(42)
infeas_arrays = np.array([np.random.choice(options, size=num_infeas, replace=True),
                          np.random.choice(options, size=num_infeas, replace=True)]).T
INFEAS_TUPLES = [tuple(x) for x in infeas_arrays]
# always exclude the other minima
INFEAS_TUPLES.append((7, 11))
INFEAS_TUPLES.append((7, 15))
INFEAS_TUPLES.append((13, 5))


def constraint(x):
    """ Evatuates the constraint """
    x0 = x[0]  # int
    x1 = x[1]  # int
    sample_tuple = (x0, x1)
    if sample_tuple in INFEAS_TUPLES:
        return False
    return True
