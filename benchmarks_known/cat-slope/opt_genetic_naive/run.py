#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import pickle
from gryffin import Gryffin
sys.path.append("../../")

from benchmark_functions import write_categories, save_pkl_file, load_data_from_pkl_and_continue
from benchmark_functions import CatSlopeConstr as BenchmarkSurface

# --------
# Settings
# --------
acquisition_optimizer = "genetic"
with_descriptors = False
dynamic = False  # use static Gryffin

budget = 441  # i.e. run all options
repeats = 100
random_seed = None  # i.e. use different random seed each time
surface = BenchmarkSurface()
sampling_strategies = np.array([1, -1])

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)
# write categories
if with_descriptors is True and dynamic is True:
    num_descs = 2  # dynamic gryffin
else:
    num_descs = 1  # naive (not used), and static
write_categories(num_dims=2, num_opts=21, home_dir='.', num_descs=num_descs, with_descriptors=with_descriptors)

# --------------------------------
# Standardized script from here on
# --------------------------------
config = {
     "general": {
             "auto_desc_gen": dynamic,
             "batches": 1,
             "num_cpus": 4,
             "boosted":  False,
             "caching": True,
             "backend": "tensorflow",
             "save_database": False,
             "sampling_strategies": len(sampling_strategies),
             "random_seed":random_seed,
             "acquisition_optimizer": acquisition_optimizer,
             "verbosity": 2  # show only warnings and errors
                },
    "parameters": [
         {"name": "x0", "type": "categorical", "size": 1, "category_details": "CatDetails/cat_details_x0.pkl"},
         {"name": "x1", "type": "categorical", "size": 1, "category_details": "CatDetails/cat_details_x1.pkl"}
    ],
    "objectives": [
        {"name": "obj", "goal": "min"}
    ]
}

for num_repeat in range(missing_repeats):
    gryffin = Gryffin(config_dict=config, known_constraints=surface.is_feasible)
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
        
        # evaluate the proposed parameters
        X = np.array([param[key] for key in gryffin.config.param_names])
        merit = np.squeeze(surface.run(X))[()]  # olympus returns 2d array
        param['obj'] = merit

        # append observation
        observations.append(param)

        # stop optimization if we found best
        if (param['x0'], param['x1']) == surface.best:
            break
    
    # store run results into a DataFrame
    x0_col = [x['x0'] for x in observations]
    x1_col = [x['x1'] for x in observations]
    obj_col = [x['obj'] for x in observations]
    data = pd.DataFrame({'x0':x0_col, 'x1':x1_col, 'obj':obj_col})
    data_all_repeats.append(data)
    
    # save results to disk
    save_pkl_file(data_all_repeats)

