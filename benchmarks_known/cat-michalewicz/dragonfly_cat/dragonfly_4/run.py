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
from benchmark_functions import CatMichalewiczConstr as BenchmarkSurface


#----------
# Settings
#----------

budget = 323
num_opts = 21
num_dims = 2
repeats = 20
random_seed = None
surface_ = BenchmarkSurface()

def surface(x):
    return surface_.run(x)


# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)


# set dragonfly parameter space
domain_var = []
for dim in range(num_dims):
    domain_var.append(
        {
            'name': f'x{dim}',
            'type': 'discrete',
            'items': [f'x_{i}' for i in range(num_opts)],
        }
    )

domain_constraints = [
    {'name': 'dc1', 'constraint': constraint}
]

config_params = {'domain': domain_var, 'domain_constraints': domain_constraints}
config = load_config(config_params)

for num_repeat in range(missing_repeats):
    print(f'===============================')
    print(f'   Repeat {len(data_all_repeats)+1}')
    print(f'===============================')

    # func_caller = CPFunctionCaller(None, config.domain, domain_orderings=config.domain_orderings)
    #
    # opt = gp_bandit.CPGPBandit(func_caller, ask_tell_mode=False)
    # opt.initialise()

    # optimize
    opt_val, opt_pt, history = minimise_function(
        surface, config.domain, budget, opt_method='bo', config=config,
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
