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
def cost_per_minute(flow_c60, flow_sultine):
    # C60 cost on Sigma Aldrich: $422 for 5 g
    # Dibromo-o-xylene cost on Sigma Aldrich: $191 for 100 g
    # C60 concentration: 2 mg/mL
    # Sultine concentration: 1.4 mg/mL
    #
    # say we consider a scale up where flows are L/min instead of uL/min
    # (relative costs are the same anyway, just nicer numbers to look at and because cost might matter more at larger scale)
    # $422 / 5000 mg x 2 mg/mL = 0.1688 $/mL = 168.8 $/L for C60
    # $191 / 100000mg x 1.4 mg/mL = 0.002674 $/mL = 2.674 $/L for Sultine

    # cost of reagents in $/min (assuming flows of L/min)

    # convert values from uL/min to L/min
    flow_c60 = 1e-6*flow_c60
    flow_sultine = 1e-6*flow_sultine
    return (flow_c60*168.8) + (flow_sultine*2.674)


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
    param['obj0'] = ma+ba # obj0 = maximize [X2]+[X1]>0.9
    param['obj1'] = cost_per_minute(
                param['c60_flow'], param['sultine_flow']
    )    # obj1 = minimize cost as much as possible

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
