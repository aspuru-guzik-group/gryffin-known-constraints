#!/usr/bin/env python

import sys
import json
import numpy as np
import pandas as pd
import pickle

from dragonfly.exd.cp_domain_utils import load_config
from dragonfly import minimise_function

from dc1_constraint import constraint

sys.path.append("../../")

from benchmark_functions import write_categories, save_pkl_file, load_data_from_pkl_and_continue
from benchmark_functions import DejongConstr as BenchmarkSurface


#---------
# Settings
#---------

budget = 100
repeats = 100
random_seed = None
surface_ = BenchmarkSurface()

def surface(x):
    return surface_.run(x)[0][0]

# check wether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)

# --------------------------------
# Standardized script from here on
# --------------------------------

# set dragonfly parameter space
domain_vars = [
    {'name': 'x0', 'type': 'float', 'min': 0, 'max': 1},
    {'name': 'x1', 'type': 'float', 'min': 0, 'max': 1},
]

domain_constraints = [
    {'name': 'dc1', 'constraint': constraint}
]

config_params = {'domain': domain_vars, 'domain_constraints': domain_constraints}
config = load_config(config_params)


for num_repeat in range(missing_repeats):
    print(f'===============================')
    print(f'   Repeat {len(data_all_repeats)+1}')
    print(f'===============================')

    # optimize
    opt_val, opt_pt, history = minimise_function(
        surface, config.domain, budget, config=config
    )

    # raw query param points --> history.query_points_raw
    # raw target values --> query_true_vals

    # store the runs in a DataFrame
    params = history.query_points_raw
    values = history.query_true_vals
    x0 = [p[0] for p in params]
    x1 = [p[1] for p in params]
    obj = values
    data = pd.DataFrame({'x0': x0, 'x1': x1, 'obj': obj})
    data_all_repeats.append(data)

    # save result in pickle file
    save_pkl_file(data_all_repeats)
