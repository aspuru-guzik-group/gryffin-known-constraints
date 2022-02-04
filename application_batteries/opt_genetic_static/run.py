#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
import os
import shutil
from gryffin import Gryffin

# --------
# Settings
# --------

acquisition_optimizer = "genetic"
budget = 100
repeats = 40
random_seed = None  # i.e. use different random seed each time
sampling_strategies = np.array([1, -1])

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


def get_descriptors(group_type, lookup_label, desc_names, desc):
    """ returns list of descriptor values for a given
    list of descriptor types
    """
    desc_vals = []
    for desc_name in desc_names:
        val = desc[desc[f'{group_type}_label']==lookup_label][f'{group_type}_{desc_name}'].tolist()[0]
        desc_vals.append(val)
    return desc_vals


# ------
# Config
# ------
DESC_NAMES_R1 = ['nHetero', 'MW', 'nBondsT', 'TopoPSA', 'nHeavyAtom', 'nRot', 'apol', 'FCSP3', 'Diameter', 'nBondsD']
DESC_NAMES_R3 = ['nHetero', 'MW', 'nBondsT', 'TopoPSA', 'nHeavyAtom', 'nRot', 'apol', 'FCSP3', 'Diameter', 'nBondsD']
DESC_NAMES_R4 = ['nHetero', 'MW', 'nBondsT', 'TopoPSA', 'nHeavyAtom', 'nRot', 'apol', 'FCSP3', 'Diameter', 'nBondsD']
DESC_NAMES_R5 = ['nHetero', 'MW', 'nBondsT', 'TopoPSA', 'nHeavyAtom', 'nRot', 'apol', 'FCSP3', 'Diameter', 'nBondsD']

R1_LABELS = {i: get_descriptors('r1', i, DESC_NAMES_R1, df_results) for i in df_results.loc[: , 'r1_label'].unique()}
R3_LABELS = {i: get_descriptors('r3', i, DESC_NAMES_R3, df_results) for i in df_results.loc[: , 'r3_label'].unique()}
R4_LABELS = {i: get_descriptors('r4', i, DESC_NAMES_R4, df_results) for i in df_results.loc[: , 'r4_label'].unique()}
R5_LABELS = {i: get_descriptors('r5', i, DESC_NAMES_R5, df_results) for i in df_results.loc[: , 'r5_label'].unique()}


config = {
     "general": {
             "num_cpus": 4,
             "auto_desc_gen": False,
             "batches": 1,
             "sampling_strategies": 1,
             "boosted":  False,
             "caching": True,
             "random_seed": random_seed,
             "acquisition_optimizer": acquisition_optimizer,
             "verbosity": 3
                },
    "parameters": [
        {"name": "r1_label", "type": "categorical", "category_details": R1_LABELS},
        {"name": "r3_label", "type": "categorical", "category_details": R3_LABELS},
        {"name": "r4_label", "type": "categorical", "category_details": R4_LABELS},
        {"name": "r5_label", "type": "categorical", "category_details": R5_LABELS}
    ],
    "objectives": [
        {"name": "abs_lam_diff", "goal": "min", "tolerance": 25.0, "absolute": True},
        {"name": "ered", "goal": "min", "tolerance": 2.0383718, "absolute": True},
        {"name": "gsol", "goal": "min", "tolerance": 0.0, "absolute": False},
    ]
}

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)


for num_repeat in range(missing_repeats):
    gryffin = Gryffin(config_dict=config, known_constraints=known_constraints)
    observations = []
    for num_iter in range(budget):
        print(f'===============================')
        print(f'   Repeat {len(data_all_repeats)+1} -- Iteration {num_iter+1}')
        print(f'===============================')

        # select alternating sampling strategy
        select_idx = num_iter % len(sampling_strategies)
        sampling_strategy = sampling_strategies[select_idx]

        # query for new parameters
        params = gryffin.recommend(
            observations=observations, sampling_strategies=[sampling_strategy]
        )

        # select the single set of params we created
        param = params[0]

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
