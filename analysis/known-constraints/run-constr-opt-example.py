#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
from gryffin import Gryffin
import pickle
from olympus.surfaces import Dejong, HyperEllipsoid

import sys
sys.path.append('../../benchmarks_unknown/')
from benchmark_functions import Constr


optimizer = "genetic"

def get_surface_instance(surface_id):
    if surface_id == '1':
        surface = DummySurface1()
    elif surface_id == '2':
        surface = DummySurface2()
    elif surface_id == '3':
        surface = DummySurface3()
    elif surface_id == '4':
        surface = DummySurface4()
    return surface

# ========
# Surfaces
# ========
class DummySurface1(HyperEllipsoid, Constr):
    def is_feasible(self, Xi):
        if isinstance(Xi, dict):
            x0 = Xi['x0']
            x1 = Xi['x1']
        else:
            x0 = Xi[0]
            x1 = Xi[1]
        if x0 + x1 > 1.0:
            return False
        else:
            return True
        
        
class DummySurface2(HyperEllipsoid, Constr):
    def is_feasible(self, Xi):
        if isinstance(Xi, dict):
            x0 = Xi['x0']
            x1 = Xi['x1']
        else:
            x0 = Xi[0]
            x1 = Xi[1]
        
        if x0**2 + (x1-1)**2 < 0.25:
            return False
        if (x0-1)**2 + x1**2 < 0.25:
            return False
        return True
    
    
class DummySurface3(HyperEllipsoid, Constr):
    def is_feasible(self, Xi):
        if isinstance(Xi, dict):
            x0 = Xi['x0']
            x1 = Xi['x1']
        else:
            x0 = Xi[0]
            x1 = Xi[1]
        Xi = np.array([x0, x1])
        
        if np.sum(np.sqrt(np.abs(Xi-0.5))) < 0.70:
            return False
        return True
    

class DummySurface4(HyperEllipsoid, Constr):
    def is_feasible(self, Xi):
        if isinstance(Xi, dict):
            x0 = Xi['x0']
            x1 = Xi['x1']
        else:
            x0 = Xi[0]
            x1 = Xi[1]
        Xi = np.array([x0, x1])
        
        centers = [[0.25, 0.25],[0.25, 0.75],[0.5, 0.5], [0.75, 0.25],[0.75, 0.75]]
        
        for center in centers:
            if (x0 - center[0])**2 + (x1 - center[1])**2 < 0.015:
                return False
        return True

# ====
# Main
# ====
config = {
     "general": {
             "batches": 1,
             "num_cpus": 1,
             "boosted":  False,
             "caching": False,
             "save_database": False,
             "sampling_strategies": 1,
             "random_seed":42,
             "acquisition_optimizer": optimizer
                },
    "parameters": [{"name": f"x0", "type": "continuous", "low": 0., "high": 1.},
                   {"name": f"x1", "type": "continuous", "low": 0., "high": 1.}],
    "objectives": [{"name": "obj", "goal": "min"}]
}

# -------------
# Run Benchmark
# -------------
max_iter = 100

for surface_id in ['1', '2', '3', '4']:
    surface = get_surface_instance(surface_id)


    gryffin = Gryffin(config_dict=config, known_constraints=surface.is_feasible)

    observations = []
    sampling_strategies = np.array([1, -1])

    for num_iter in range(max_iter):
        print('ITERATION: ', num_iter+1)
    
        select_idx = num_iter % len(sampling_strategies)
        sampling_strategy = sampling_strategies[select_idx]
    
        # query for new parameters
        params = gryffin.recommend(observations=observations, sampling_strategies=[sampling_strategy])
        param = params[0]
    
        # evaluate the proposed parameters
        observation = surface.eval_merit(param)
        observations.append(observation)


    with open(f'observations-surface_{surface_id}-{optimizer}.pkl', 'wb') as content:
        pickle.dump(observations, content)

