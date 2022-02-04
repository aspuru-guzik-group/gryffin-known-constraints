#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
import os
import shutil
from gryffin import Gryffin


#----------
# Settings
#----------

budget = 959
repeats = 100
RA_CUTOFF = 0.9
RA_TYPE   = 'ra_nn'  # ra_xgb or ra_nn

#----------------------
# load tabular results
#----------------------
df_results = pd.read_csv('../data/df_results.csv')
# apply RA cutoff
df_results = df_results[df_results[RA_TYPE]>=RA_CUTOFF]
print(f'\n>>> USING {RA_TYPE} CUTOFF AT {RA_CUTOFF} : {df_results.shape}\n')


def run_experiment(param):
    match = df_results.loc[
                (df_results['r1_label'] == param['r1_label']) &
                (df_results['r3_label'] == param['r3_label']) &
                (df_results['r4_label'] == param['r4_label']) &
                (df_results['r5_label'] == param['r5_label'])
            ]
    assert len(match) == 1
    ered = match.loc[:, 'ered'].to_numpy()[0]
    gsol = match.loc[:, 'gsol'].to_numpy()[0]
    abs_lam_diff = match.loc[:, 'abs_lam_diff'].to_numpy()[0]
    return ered, gsol, abs_lam_diff


def eval_merit(param):
    ered, gsol, abs_lam_diff = run_experiment(param)
    param['ered'] = ered
    param['gsol'] = gsol
    param['abs_lam_diff'] = abs_lam_diff
    return param


def known_constraints(param):
    match = df_results.loc[
                (df_results['r1_label'] == param['r1_label']) &
                (df_results['r3_label'] == param['r3_label']) &
                (df_results['r4_label'] == param['r4_label']) &
                (df_results['r5_label'] == param['r5_label'])
            ]
    assert len(match) <= 1

    if len(match) == 0:
        return False
    else:
        return True


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



#--------
# Config
#--------

def build_set_of_feasible_samples():
    group_idxs = df_results.loc[:, ['r1_label', 'r3_label', 'r4_label', 'r5_label']].to_numpy()
    np.random.shuffle(group_idxs)
    return group_idxs


# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)

for num_repeat in range(missing_repeats):
    print(f'   Repeat {len(data_all_repeats)+1}')
    samples = build_set_of_feasible_samples()
    observations = []
    for sample in samples[:budget]:

        param = {}
        param['r1_label'] = sample[0]
        param['r3_label'] = sample[1]
        param['r4_label'] = sample[2]
        param['r5_label'] = sample[3]

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
