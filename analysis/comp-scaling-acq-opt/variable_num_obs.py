#!/usr/bin/env python

import numpy as np
import pandas as pd
import seaborn as sns
from gryffin import Gryffin
import pickle
import os


constrained = True
optimizer = "adam"

dim = 2


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
    X = [params[key] for key in params.keys()]
    if np.sum(X) > 1.0:
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
             "random_seed":42,
             "acquisition_optimizer": optimizer
                },
    "parameters": parameters,
    "objectives": [{"name": "obj", "goal": "min"}]
}


if constrained is True:
    gryffin = Gryffin(config_dict=config, known_constraints=known_constraints)
else:
    gryffin = Gryffin(config_dict=config, known_constraints=None)

# -------------
# Run Benchmark
# -------------
if constrained is True:
    constr_str = 'constrained'
else:
    constr_str = 'unconstrained'


if os.path.exists(f'num_obs-{constr_str}-{optimizer}.pkl'):
    with open(f'num_obs-{constr_str}-{optimizer}.pkl', 'rb') as content:
        timings = pickle.load(content)
else:
    timings = {}


for n_obs in [10, 100, 200, 300, 400, 500]:

    if n_obs in timings.keys():
        continue

    np.random.seed(42)
    samples = gryffin.random_sampler.draw(num=n_obs)
    observations = build_observations(samples)

    # store times with info on where they came from
    timings[n_obs] = {}
    timings[n_obs][-1] = []
    timings[n_obs][0] = []
    timings[n_obs][1] = []

    # 20 x 3 = 60 samples
    for repeat in range(20):
        for strategy in [-1, 0 , 1]:
            print(f"===== {n_obs} observations / repeat No. {repeat} / stretegy [ {strategy} ] =====")
            _ = gryffin.recommend(observations=observations, sampling_strategies=[strategy])
            t = gryffin.timings['Acquisition']['proposals_opt']
            timings[n_obs][strategy].append(t)


with open(f'num_obs-{constr_str}-{optimizer}.pkl', 'wb') as content:
    pickle.dump(timings, content)

