#!/usr/bin/env python

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
import shutil
import subprocess


from dragonfly.exd.cp_domain_utils import load_config
# from dragonfly.exd import domains
from dragonfly.exd.experiment_caller import CPFunctionCaller, EuclideanFunctionCaller
from dragonfly import maximise_function, minimise_function
from dragonfly.opt import gp_bandit
#from dragonfly.parse.config_parser import load_parameters

from dc1_constraint import constraint
from chimera import Chimera

sys.path.append("../../../benchmarks_known/")

from benchmark_functions import write_categories, save_pkl_file, load_data_from_pkl_and_continue


#----------
# Settings
#----------

budget = 100
repeats = 50
num_init_samples = 10 # number of samples to draw randomly
random_seed = None


tolerances = np.array([0.9, 0.0])
absolutes = [True, True]
goals = ['max', 'min']
chimera = Chimera(
    tolerances=tolerances,
    absolutes=absolutes,
    goals=goals,
    softness=1e-3,
)


#-----------------
# Helper functions
#-----------------

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


def build_set_of_feasible_samples(n=100):
    samples = []
    while len(samples) < n:
        c60 = np.random.uniform(low=0., high=200.)
        sul = np.random.uniform(low=0., high=200.)
        T = np.random.uniform(low=100., high=150.)
        #param = {'c60_flow': c60, 'sultine_flow': sul, 'T':T}
        param = [c60, sul, T]
        if constraint(param) is True:
            samples.append(param)
    return samples



#------------
# Experiment
#------------

# set the dragonfly parameter space
domain_vars = [
    {'name': 'c60_flow', 'type': 'float', 'min': 0., 'max': 200.},
    {'name': 'sultine_flow', 'type': 'float', 'min': 0., 'max': 200.},
    {'name': 'T', 'type': 'float', 'min': 100., 'max': 150.},
]

domain_constraints = [
    {'name': 'dc1', 'constraint': constraint}
]

config_params = {'domain': domain_vars, 'domain_constraints': domain_constraints}
config = load_config(config_params)


# check wether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)

for num_repeat in range(missing_repeats):

    # get the set of random samples
    random_samples = build_set_of_feasible_samples(n=num_init_samples)

    # init Dragonfly function caller
    func_caller = EuclideanFunctionCaller(None, config.domain.list_of_domains[0], config=config)

    opt = gp_bandit.EuclideanGPBandit(func_caller, ask_tell_mode=True)
    opt.initialise()

    print(config)
    print(config.domain.list_of_domains[0])

#    quit()


    xs = []
    ys = []
    ys_all = []
    for iter_num in range(budget):

        print(f'==========================================')
        print(f'   Repeat {len(data_all_repeats)+1}\t Iter {iter_num+1}')
        print(f'==========================================')

        if iter_num < num_init_samples:
            # take the sample from the list of random samples and add
            x = random_samples[iter_num]
        else:
            # ask for a suggestion from dragonfly
            x = opt.ask()

        # EMULATOR MEASUREMENT

        # put x into the emulator form
        param  = {'c60_flow': x[0], 'sultine_flow': x[1], 'T': x[2]}

        # run the emulator in a subprocess
        with open('param.pkl', 'wb') as content:
            pickle.dump(param, content)

        subprocess.call('../emulator.py', shell=True)

        # get the results
        with open('observation.pkl', 'rb') as content:
            observation = pickle.load(content)
        os.remove('observation.pkl')

        y = [observation['obj0'], observation['obj1']]
        # include the MA and BA keys as well
        y_all = [observation['obj0'], observation['obj1'],
                observation['MA'], observation['BA']]

        xs.append(x)
        ys.append(y)
        ys_all.append(y_all)

        params = np.array(xs)
        measurements = np.array(ys)
        measurements_all = np.array(ys_all)

        merits = chimera.scalarize(measurements).reshape((params.shape[0], 1))



        print(xs)
        print(ys)
        print(ys_all)

        print(params.shape)
        print(measurements.shape)
        print(measurements_all.shape)
        print(merits.shape)


        # intialize new Dragonfly object and rebuild the historical measurements
        opt = gp_bandit.EuclideanGPBandit(func_caller, ask_tell_mode=True)
        opt.initialise()
        for xi, yi in zip(xs, merits):
            yi = -yi[0]  # by default the objective is maximized, need to minimize merit
            if np.isnan(yi):
                yi = 0.
            yi = np.nan
            opt.tell([(xi, yi)]) # yi must be a scalar

        #quit()
