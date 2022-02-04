#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
from gryffin import Gryffin
import pickle
import os


constrained = False
optimizer = "genetic"

# =========
# Functions
# =========
def objective_func(X):
    # any dimension
    X = np.array(X)
    y = []
    for Xi in X:
        yi = np.sum((Xi - 0.5)**2)
        y.append(yi)
    return np.array(y)


def known_constraints(params):
    X = np.array([params[key] for key in params.keys()])
    dim = len(X)
    if np.sum(X) > 0.5 * dim:
        return False
    return True


def eval_known_constraints(X):
    X = np.array(X)
    y = []
    for Xi in X:
        params = {}
        for j, Xij in enumerate(Xi):
            params[f'x{j}'] = Xij
        yi = known_constraints(params)
        y.append(yi)
    return np.array(y)


def merit_function(param):
    X = [param[key] for key in param.keys()]
    param['obj'] = objective_func([X])[0]
    return param


def build_observations(samples):
    observations = []
    for sample in samples:
        d = {}
        for i, xi in enumerate(sample):
            d[f'x{i}'] = xi
        d = merit_function(d)
        observations.append(d)
    return observations


# ====
# Main
# ====

def make_config(dim):
    parameters = []
    for i in range(dim):
        param = {"name": f"x{i}", "type": "continuous", "low": 0., "high": 1.}
        parameters.append(param)
    

    config = {
        "general": {
             "batches": 1,
             "num_cpus": 1,
             "boosted":  False,
             "caching": False,
             "save_database": False,
             "sampling_strategies": 1,
             "random_seed":None,
             "acquisition_optimizer": optimizer
                },
        "parameters": parameters,
        "objectives": [{"name": "obj", "goal": "min"}]
    }

    return config




# -------------
# Run Benchmark
# -------------

if constrained is True:
    constr_str = 'constrained'
else:
    constr_str = 'unconstrained'

if os.path.exists(f'dims-{constr_str}-{optimizer}-increasing-obs.pkl'):
    with open(f'dims-{constr_str}-{optimizer}-increasing-obs.pkl', 'rb') as content:
        timings = pickle.load(content)
else:
    timings = {}


for dim in [1, 2, 3, 4, 6, 8]:

    if dim in timings.keys():
        continue

    n_obs = int(round(10**np.sqrt(dim), 0))

    config = make_config(dim=dim)

    # store times with info on where they came from
    timings[dim] = {}
    timings[dim][-1] = []
    timings[dim][0] = []
    timings[dim][1] = []

    # 20 x 3 = 60 samples
    for repeat in range(20):
        for strategy in [-1, 0 , 1]:

            np.random.seed(dim * 100 + repeat)
            if constrained is True:
                gryffin = Gryffin(config_dict=config, known_constraints=known_constraints)
                samples = gryffin.random_sampler.draw(num=n_obs)
            else:
                gryffin = Gryffin(config_dict=config, known_constraints=None)
                samples = gryffin.random_sampler.draw(num=n_obs)

            observations = build_observations(samples)

            print(f"===== {dim} dimensions / repeat No. {repeat} / stretegy [ {strategy} ] =====")
            _ = gryffin.recommend(observations=observations, sampling_strategies=[strategy])
            t = gryffin.timings['Acquisition']['proposals_opt']
            timings[dim][strategy].append(t)


with open(f'dims-{constr_str}-{optimizer}-increasing-obs.pkl', 'wb') as content:
    pickle.dump(timings, content)

