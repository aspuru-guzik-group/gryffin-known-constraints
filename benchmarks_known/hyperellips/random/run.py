#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import pickle
from gryffin import Gryffin
sys.path.append("../../")

from benchmark_functions import write_categories, save_pkl_file, load_data_from_pkl_and_continue
from benchmark_functions import HyperEllipsoidConstr as BenchmarkSurface

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
def build_set_of_feasible_samples(surface, n=100):
    np.random.seed(None)
    samples = []
    while len(samples) < n:
        a = list(np.random.uniform(low=0., high=1., size=(1, 2))[0])
        d = {'x0':a[0], 'x1':a[1]}
        feas = surface.eval_constr(d)
        if feas is True:
            samples.append(a)            
    return samples


for num_repeat in range(missing_repeats):

    print(f'   Repeat {len(data_all_repeats)+1}')

    X_samples = build_set_of_feasible_samples(surface=BenchmarkSurface(), n=budget)
    y_samples = []

    for X in X_samples:
        
        # evaluate the proposed parameters
        merit = np.squeeze(surface.run(X))[()]  # olympus returns 2d array
        y_samples.append(merit)

    # store run results into a DataFrame
    x0_col = [x[0] for x in X_samples]
    x1_col = [x[1] for x in X_samples]
    obj_col = y_samples
    data = pd.DataFrame({'x0':x0_col, 'x1':x1_col, 'obj':obj_col})
    data_all_repeats.append(data)
    
    # save results to disk
    save_pkl_file(data_all_repeats)

