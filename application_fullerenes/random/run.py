#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
import os, sys
import shutil

sys.path.append("../")
import torch
from emulator import BayesianNetwork


# --------
# Settings
# --------

budget = 100
repeats = 100 #200

# --------------------
# load BNN emulator model
# --------------------

device = 'cpu'
model = BayesianNetwork(3, 4, 64).to(device)
checkpoint = '../torch_prod_models/fullerenes.pth'
model.load_state_dict(torch.load(checkpoint))
# load feature scaler
feature_scaler = pickle.load(open('../torch_prod_models/feature_scaler.pkl', 'rb'))

# ---------
# Functions
# ---------
def run_experiment(param):
    c60 = param['c60_flow']
    sul = param['sultine_flow']
    T = param['T']

    x = np.array([[c60, sul, T]])
    _x  = feature_scaler.transform(x)
    pred, _ = model.predict(torch.tensor(_x).float())
    na, ma, ba, ta = pred.cpu().detach().numpy()[0]

    return na, ma, ba, ta

def eval_merit(param):
    na, ma, ba, ta = run_experiment(param)
    param['obj0'] = ba / ma  # obj0 = max MA and BA
    param['obj1'] = ta   # obj1 = min TA
    #param['obj2'] = ma   # obj2 = max BA

    # append also individual fractions
    param['NA'] = na
    param['MA'] = ma
    param['BA'] = ba
    param['TA'] = ta
    return param

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


for num_repeat in range(missing_repeats):
    print(f'   Repeat {len(data_all_repeats)+1}')

    samples = build_set_of_feasible_samples(budget)
    observations = []

    for param in samples:

        # evaluate the proposed parameters
        observation = eval_merit(param)

        # append observation
        observations.append(observation)

    # store run results into a DataFrame
    data_dict = {}
    for key in observations[0].keys():
        data_dict[key] = [o[key] for o in observations]
    data = pd.DataFrame(data_dict)
    data_all_repeats.append(data)

    # save results to disk
    save_pkl_file(data_all_repeats)
