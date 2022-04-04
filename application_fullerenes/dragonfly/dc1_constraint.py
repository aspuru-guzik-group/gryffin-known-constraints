'''
fullerenes example constraint for Dragonfly

def known_constraints(param):
    flow_sum = param['c60_flow'] + param['sultine_flow']
    b0 = 30. < flow_sum < 310.
    b1 = np.logical_and(param['c60_flow'] > 0.5 * param['sultine_flow'],
                        param['c60_flow'] < 2. * param['sultine_flow'])

    if np.logical_and(b0, b1) == True:
        return True
    else:
        return False


'''

import numpy as np

def constraint(x):
    ''' evaluates the constraint '''
    flow_sum = x[0] + x[1]
    b0 = 30. < flow_sum < 310.
    b1 = np.logical_and(
            x[0] > 0.5 * x[1],
            x[0] < 2. * x[1]
        )
    if np.logical_and(b0, b1) == True:
        return True
    else:
        return False
