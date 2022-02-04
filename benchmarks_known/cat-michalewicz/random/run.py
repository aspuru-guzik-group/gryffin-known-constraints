#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import pickle
from gryffin import Gryffin
sys.path.append("../../")

from benchmark_functions import write_categories, save_pkl_file, load_data_from_pkl_and_continue
from benchmark_functions import CatMichalewiczConstr as BenchmarkSurface

# --------
# Settings
# --------
budget = 100
repeats = 100
random_seed = None  # i.e. use different random seed each time
surface = BenchmarkSurface()
sampling_strategies = np.array([1, -1])

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)

# --------------------------------
# Standardized script from here on
# --------------------------------
def build_set_of_feasible_samples(surface):
    samples = []
    domain = np.arange(surface.num_opts)
    Z = np.zeros((surface.num_opts, surface.num_opts))
    for x_index, x in enumerate(domain):
        for y_index, y in enumerate(domain):
            x = ['x_{}'.format(x_index), 'x_{}'.format(y_index)]
            feas = surface.eval_constr({'x0': x[0], 'x1': x[1]})
            if feas is True:
                samples.append(x)       
    return samples


for num_repeat in range(missing_repeats):

    print(f'   Repeat {len(data_all_repeats)+1}')

    X_samples_all = build_set_of_feasible_samples(surface=BenchmarkSurface())
    np.random.shuffle(X_samples_all)

    X_samples = []
    y_samples = []

    for X in X_samples_all:
        
        # evaluate the proposed parameters
        merit = np.squeeze(surface.run(X))[()]  # olympus returns 2d array

        X_samples.append(X)
        y_samples.append(merit)

        # stop optimization if we found best
        if (X[0], X[1]) == surface.best:
            break

    # store run results into a DataFrame
    x0_col = [x[0] for x in X_samples]
    x1_col = [x[1] for x in X_samples]
    obj_col = y_samples
    data = pd.DataFrame({'x0':x0_col, 'x1':x1_col, 'obj':obj_col})
    data_all_repeats.append(data)
    
    # save results to disk
    save_pkl_file(data_all_repeats)

