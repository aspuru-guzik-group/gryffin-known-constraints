#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
import os, sys
import shutil
import subprocess

import GPy
import GPyOpt

from chimera import Chimera

# sys.path.append("../")
# import torch
# from emulator import BayesianNetwork


# --------
# Settings
# --------

budget = 100
repeats = 100
n_init_samples = 5

tolerances = np.array([1.0, 0.0])
absolutes = [True, True]
goals = ['max', 'min']
chimera = Chimera(
    tolerances=tolerances,
    absolutes=absolutes,
    goals=goals,
    softness=1e-3,
)

# --------------------
# load BNN emulator model
# --------------------

# device = 'cpu'
# model = BayesianNetwork(3, 4, 64).to(device)
# checkpoint = '../torch_prod_models/fullerenes.pth'
# model.load_state_dict(torch.load(checkpoint))
# # load feature scaler
# feature_scaler = pickle.load(open('../torch_prod_models/feature_scaler.pkl', 'rb'))

# ---------
# Functions
# ---------

# def run_experiment(param):
#     # param is 2d array
#     c60 = param[0, 0]
#     sul = param[0, 1]
#     T = param[0, 2]
#
#     x = np.array([[c60, sul, T]])
#     _x  = feature_scaler.transform(x)
#     pred, _ = model.predict(torch.tensor(_x).float())
#     na, ma, ba, ta = pred.cpu().detach().numpy()[0]
#
#     return np.array([na, ma, ba, ta])


def known_constraints(param):
    flow_sum = param['c60_flow'] + param['sultine_flow']
    b0 = 30. < flow_sum < 310.
    b1 = np.logical_and(param['c60_flow'] > 0.5 * param['sultine_flow'],
                        param['c60_flow'] < 2. * param['sultine_flow'])

    if np.logical_and(b0, b1) == True:
        return True
    else:
        return False


def save_pkl_file(data_all_repeats):
    """save pickle file with results so far"""

    if os.path.isfile('results.pkl'):
        shutil.move('results.pkl', 'bkp-results.pkl')  # overrides existing files

    # store run results to disk
    with open("results.pkl", "wb") as content:
        pickle.dump(data_all_repeats, content)



def load_data_from_pkl_and_continue(N):
    """load results from pickle file"""

    data_all_repeats = []
    # if no file, then we start from scratch/beginning
    if not os.path.isfile('results.pkl'):
        return data_all_repeats, N

    # else, we load previous results and continue
    with open("results.pkl", "rb") as content:
        data_all_repeats = pickle.load(content)

    missing_N = N - len(data_all_repeats)

    return data_all_repeats, missing_N


def build_set_of_feasible_samples(n=100):
    samples = []
    while len(samples) < n:
        c60 = np.random.uniform(low=0., high=200.)
        sul = np.random.uniform(low=0., high=200.)
        T = np.random.uniform(low=100., high=150.)
        param = {'c60_flow': c60, 'sultine_flow': sul, 'T':T}
        if known_constraints(param) is True:
            samples.append(param)
    return samples



# ------
# Main
# ------

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)


# space for GPyOpt
space = [
    {'name': 'c60_flow', 'type': 'continuous', 'domain': (0.0, 200.0)},
    {'name': 'sultine_flow', 'type': 'continuous', 'domain': (0.0, 200.0)},
    {'name': 'T', 'type': 'continuous', 'domain': (100.0, 150.0)},

]

# constraints for GPyOpt --> the constraint expression must
# evalaute to be less than or equal to 0 to statisfy that specific
# constraint. Feasible points satisfy all constraint expressions
# x is a 2d array, second index is
# 0 --> c60_flow
# 1 --> sultine_flow
# 2 --> T
constraints = [
    {'name': 'constr_1', 'constraint': 'x[:, 0] + x[:, 1] - 310.0' },
    {'name': 'constr_2', 'constraint': '10.0 - x[:, 0] - x[:, 1]' },
    {'name': 'constr_3', 'constraint': '0.5*x[:, 1] - x[:, 0]' },
    {'name': 'constr_4', 'constraint': 'x[:, 0] - 2*x[:, 1]' },
]

feasible_region = GPyOpt.Design_space(space=space, constraints=constraints)


for num_repeat in range(missing_repeats):
    print(f'   Repeat {len(data_all_repeats)+1}')

    # build the set of feasible samples for the initial design
    samples = build_set_of_feasible_samples(budget)
    # randomly shuffle the samples
    np.random.shuffle(samples)
    # take the first for the initial design
    x_init = []
    y_init = []
    y_init_all = []
    for s_ix in range(n_init_samples):
        x = [samples[s_ix]['c60_flow'], samples[s_ix]['sultine_flow'], samples[s_ix]['T']]

        # put x in the form needed for the emulator
        param = {'c60_flow': x[0], 'sultine_flow': x[1], 'T': x[2]}

        # run the emulator in a subprocess
        with open('param.pkl', 'wb') as content:
            pickle.dump(param, content)

        subprocess.call('../emulator.py', shell=True)

        # get the results
        with open('observation.pkl', 'rb') as content:
            observation = pickle.load(content)
        os.remove('observation.pkl')

        y = np.array([observation['obj0'], observation['obj1']])
        # include the MA and BA keys as well
        y_all = np.array(
            [observation['obj0'], observation['obj1'],
            observation['MA'], observation['BA']]
        )

        x_init.append(x)
        y_init.append(y)
        y_init_all.append(y_all)

    params = np.array(x_init)
    measurements = np.array(y_init)
    measurements_all = np.array(y_init_all)

    merits = chimera.scalarize(measurements).reshape((params.shape[0], 1))

    while params.shape[0] < budget:

        bo_step = GPyOpt.methods.BayesianOptimization(
            f=None, domain=space, constraints=constraints, X=params, Y=merits
        )
        x_next = bo_step.suggest_next_locations()

        # put x_next in form needed for the emulator
        param = {'c60_flow': x_next[0][0], 'sultine_flow': x_next[0][1], 'T': x_next[0][2]}

        # run the emulator in suprocess
        with open('param.pkl', 'wb') as content:
            pickle.dump(param, content)

        subprocess.call('../emulator.py', shell=True)

        # get the results
        with open('observation.pkl', 'rb') as content:
            observation = pickle.load(content)
        os.remove('observation.pkl')

        y_next = np.array([observation['obj0'], observation['obj1']])

        y_next_all =  np.array(
            [observation['obj0'], observation['obj1'],
             observation['MA'], observation['BA']]
        )

        params = np.vstack((params, x_next))
        measurements = np.vstack((measurements, y_next))
        measurements_all = np.vstack([measurements_all, y_next_all])

        # re-compute merits
        merits = chimera.scalarize(measurements).reshape((params.shape[0], 1))


    # store run results into a DataFrame
    params_keys = ['c60_flow', 'sultine_flow', 'T']
    obj_keys    = ['obj0', 'obj1', 'MA', 'BA']
    data_dict = {}

    for p_ix, p_key in enumerate(params_keys):
        data_dict[p_key] = params[:, p_ix]
    for o_ix, o_key in enumerate(obj_keys):
        data_dict[o_key] = measurements_all[:, o_ix]
    data = pd.DataFrame(data_dict)
    data_all_repeats.append(data)

    # save results to disk
    save_pkl_file(data_all_repeats)
