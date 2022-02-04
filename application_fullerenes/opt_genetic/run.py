#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
import os
import shutil
from gryffin import Gryffin
import subprocess

# --------
# Settings
# --------

acquisition_optimizer = "genetic"
budget = 100
repeats = 100
random_seed = None  # i.e. use different random seed each time
sampling_strategies = np.array([1, -1])

# ---------
# Functions
# ---------
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

# ------
# Config
# ------
config = {
     "general": {
             "num_cpus": 4,
             "auto_desc_gen": False,
             "batches": 1,
             "sampling_strategies": 1,
             "boosted":  False,
             "caching": False,
             "random_seed": random_seed,
             "acquisition_optimizer": acquisition_optimizer,
             "verbosity": 3
                },
    "parameters": [
        {"name": "c60_flow", "type": "continuous", "low": 0., "high": 200.},
        {"name": "sultine_flow", "type": "continuous", "low": 0., "high": 200.},
        {"name": "T", "type": "continuous", "low": 100., "high": 150.}
    ],
    "objectives": [
        {"name": "obj0", "goal": "max", "tolerance": 1.0, "absolute": True},
        {"name": "obj1", "goal": "min", "tolerance": 0.0, "absolute": True},
        #{"name": "obj2", "goal": "max", "tolerance": 1., "absolute": True}
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
        params = gryffin.recommend(observations=observations, sampling_strategies=[sampling_strategy])

        # select the single set of params we created
        param = params[0]

        # run emulator in a subprocess
        with open('param.pkl', 'wb') as content:
            pickle.dump(param, content)

        subprocess.call("../emulator.py", shell=True)

        # get the results
        with open('observation.pkl', 'rb') as content:
            observation = pickle.load(content)
        os.remove('observation.pkl')

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
