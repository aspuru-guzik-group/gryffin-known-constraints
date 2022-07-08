#!/usr/bin/env python

import os
import pickle
import shutil
import numpy as np
from copy import deepcopy
from olympus.surfaces import Dejong, Branin, HyperEllipsoid, AckleyPath, Levy, Michalewicz, Rastrigin, Schwefel, StyblinskiTang
from olympus.surfaces import DiscreteAckley, DiscreteDoubleWell, DiscreteMichalewicz, LinearFunnel, NarrowFunnel


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


# function to write category details/descriptors
def write_categories(
    num_dims=2,
    num_opts=21,
    home_dir='.',
    num_descs=1,
    with_descriptors=True,
    surface=None,
):
    #check to see if we have a permutation mapping
    if surface:
        # assume we have a mapping
        permut_map = lambda x : surface.permut_map[x]
    else:
        permut_map = lambda x : x
    for dim in range(num_dims):
        cat_details = {}
        for opt in range(num_opts):
            if with_descriptors is True:
                desc = round(float(permut_map(f'x_{opt}')[2:]))
                cat_details[f'x_{opt}'] = [desc] * num_descs
            else:
                cat_details[f'x_{opt}'] = None

        # create cat_details dir if necessary
        if not os.path.isdir('%s/CatDetails' % home_dir):
            os.mkdir('%s/CatDetails' % home_dir)

        cat_details_file = f'{home_dir}/CatDetails/cat_details_x{dim}.pkl'
        pickle.dump(cat_details, open(cat_details_file, 'wb'))


# ==============
# Parent classes
# ==============

# --------------------------------------
# Parent classes for Constrained surface
# --------------------------------------
class Constr:
    def eval_constr(self, params):
        """Evaluate constraints, return True for feasible locations
        and False for infeasible ones.
        """
        if isinstance(params, list):
            Y = []
            for param in params:
                Y.append(self.is_feasible(param))
            return np.array(Y)

        elif isinstance(params, dict):
            return self.is_feasible(params)

    def run_constr(self, X):
        "Evaluate surface and return nan for infeasible locations"
        if len(np.shape(X)) == 2:
            Y = []
            for Xi in X:
                if self.is_feasible(Xi) is True:
                    Yi = self.run(Xi)
                    Y.append(np.squeeze(Yi))
                else:
                    Y.append(np.nan)
            return np.array(Y)
        elif len(np.shape(X)) == 1:
            if self.is_feasible(X) is True:
                return np.squeeze(self.run(X))[()]
            else:
                return np.nan

    def eval_merit(self, param):
        """Evaluate merit of Gryffin's param object.
        """
        x0 = param['x0']
        x1 = param['x1']
        param['obj'] = self.run_constr([x0, x1])
        return param

# -------------------------------------
# Parent class for Categorical surfaces
# -------------------------------------
class CategoricalEvaluator:

    def __init__(self, num_dims=2, num_opts=21):
        self.num_dims = num_dims
        self.num_opts = num_opts

    @staticmethod
    def str2array(sample):
        return np.array([round(float(entry[2:])) for entry in np.squeeze(sample)])

    def run(self, sample):
        vector = self.str2array(sample=sample)
        return self.evaluate(sample=vector)

# ==============================
# Constrained benchmark surfaces
# ==============================
class DejongConstr(Dejong, Constr):
    def is_feasible(self, params):
        x0 = params['x0']
        x1 = params['x1']
        y = (x0-0.5)**2 + (x1-0.5)**2

        if np.abs(x0-x1) < 0.1:
            return False

        if 0.05 < y < 0.15:
            return False
        else:
            return True

class BraninConstr(Branin, Constr):
    def is_feasible(self, params):
        x0 = params['x0']
        x1 = params['x1']

        y0 = (x0-0.12389382)**2 + (x1-0.81833333)**2
        y1 = (x0-0.961652)**2 + (x1-0.165)**2

        if y0 < 0.2**2 or y1 < 0.35**2:
            return False
        else:
            return True

class StyblinskiTangConstr(StyblinskiTang, Constr):
    def is_feasible(self, params):
        x0 = params['x0']
        x1 = params['x1']

        if x0+x1 < 0.4:
            return False
        if x0 > 0.6 and x1 > 0.6:
            return False
        if x0 < 0.4 and x1 > 0.6:
            return False
        if x0 > 0.6 and x1 < 0.4:
            return False
        return True


class HyperEllipsoidConstr(HyperEllipsoid, Constr):
    def __init__(self):
        HyperEllipsoid.__init__(self)

        np.random.seed(42)
        N = 20
        self.centers = [np.random.uniform(low=0.0, high=1.0, size=2) for i in range(N)]
        self.radii = [np.random.uniform(low=0.05, high=0.15, size=1) for i in range(N)]

    def is_feasible(self, params):
        x0 = params['x0']
        x1 = params['x1']
        Xi = np.array([x0, x1])
        for c, r in zip(self.centers, self.radii):
            if np.linalg.norm(c - Xi) < r:
                return False

        return True


class SchwefelConstr(Schwefel, Constr):
	def __init__(self):
		Schwefel.__init__(self)

		np.random.seed(42)
		N = 20
		self.centers = [np.random.uniform(low=0.0, high=1.0, size=2) for i in range(N)]
		self.radii = [np.random.uniform(low=0.05, high=0.15, size=1) for i in range(N)]

	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		Xi = np.array([x0, x1])
		for c, r in zip(self.centers, self.radii):
			if np.linalg.norm(c - Xi) < r:
				return False
		return True

class DiscreteAckleyConstr(DiscreteAckley, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		if x0+x1 < 0.3:
			return False
		return True

# ====================
# Categorical surfaces
# ====================
class CatDejongConstr(CategoricalEvaluator, Constr):
    """
        Dejong is to be evaluated on the hypercube
        x_i in [-5.12, 5.12] for i = 1, ..., d
    """
    def dejong(self, vector):
        result = np.sum(vector**2)
        return result

    def evaluate(self, sample):
        # map sample onto hypercube
        vector = np.zeros(self.num_dims)
        for index, element in enumerate(sample):
            vector[index] = 10.24 * ( element / float(self.num_opts - 1) ) - 5.12
        return self.dejong(vector)

    def is_feasible(self, params):
        x0 = params['x0']  # str
        x1 = params['x1']  # str
        Xi = self.str2array([x0, x1])
        x0 = Xi[0]  # float
        x1 = Xi[1]  # float
        if x0 in [9, 11]:
            return False
        if x1 in [9, 11]:
            return False
        return True

    @property
    def best(self):
        return ('x_10', 'x_10')


class CatSlopeConstr(CategoricalEvaluator, Constr):
    """
        Response sampled from standard normal distribution
        with correlation
    """
    def random_correlated(self, vector):
        seed   = 0
        vector = np.array(vector)
        for index, element in enumerate(vector):
            seed += self.num_opts**index * element
        result = np.sum(vector / self.num_opts)
        return result

    def evaluate(self, sample):
        return self.random_correlated(sample)

    def is_feasible(self, params):
        x0 = params['x0']  # str
        x1 = params['x1']  # str
        Xi = self.str2array([x0, x1])
        x0 = Xi[0]  # float
        x1 = Xi[1]  # float

        y = x0**2 + x1**2
        if 5 < y < 25:
            return False
        if 70 < y < 110:
            return False
        if 200 < y < 300:
            return False
        return True

    @property
    def best(self):
        return ('x_0', 'x_0')


class CatMichalewiczConstr(CategoricalEvaluator, Constr):
    """
        Michalewicz is to be evaluated on the hypercube
        x_i in [0, pi] for i = 1, ..., d
    """

    def michalewicz(self, vector, m=10.):
        result = 0.
        for index, element in enumerate(vector):
            result += - np.sin(element) * np.sin( (index + 1) * element**2 / np.pi)**(2 * m)
        return result

    def evaluate(self, sample):
        # map sample onto hypercube
        vector = np.zeros(self.num_dims)
        for index, element in enumerate(sample):
            vector[index] = np.pi * element / float(self.num_opts - 1)
        return self.michalewicz(vector)

    def is_feasible(self, params):
        x0 = params['x0']  # str
        x1 = params['x1']  # str
        Xi = self.str2array([x0, x1])
        x0 = Xi[0]  # float
        x1 = Xi[1]  # float

        y = ((x0-14))**2 + (x1-10)**2
        if 5 < y < 30:
            return False
        if 12.5 < x0 < 15.5:
            if x1 < 5.5:
                return False
        if 8.5 < x1 < 11.5:
            if x0 < 9.5:
                return False
        return True

    @property
    def best(self):
        return ('x_14', 'x_10')


class CatCamelConstr(CategoricalEvaluator, Constr):
    """
        Camel is to be evaluated on the hypercube
        x_i in [-3, 3] for i = 1, ..., d
    """
    def __init__(self, num_dims=2, num_opts=21):
        CategoricalEvaluator.__init__(self, num_dims, num_opts)

        # choose infeasible points at random
        options = [f'x_{i}' for i in range(0, num_opts, 1)]
        num_infeas = 100
        np.random.seed(42)
        infeas_arrays = np.array([np.random.choice(options, size=num_infeas, replace=True),
                                  np.random.choice(options, size=num_infeas, replace=True)]).T
        self.infeas_tuples = [tuple(x) for x in infeas_arrays]
        # always exclude the other minima
        self.infeas_tuples.append(('x_7', 'x_11'))
        self.infeas_tuples.append(('x_7', 'x_15'))
        self.infeas_tuples.append(('x_13', 'x_5'))

    def camel(self, vector):
        result = 0.

        # global minima
        loc_0 = np.array([-1., 0.])
        loc_1 = np.array([ 1., 0.])
        weight_0 = np.array([4., 1.])
        weight_1 = np.array([4., 1.])

        # local minima
        loc_2  = np.array([-1., 1.5])
        loc_3  = np.array([ 1., -1.5])
        loc_5  = np.array([-0.5, -1.0])
        loc_6  = np.array([ 0.5,  1.0])
        loss_0 = np.sum(weight_0 * (vector - loc_0)**2) + 0.01 + np.prod(vector - loc_0)
        loss_1 = np.sum(weight_1 * (vector - loc_1)**2) + 0.01 + np.prod(vector - loc_1)
        loss_2 = np.sum((vector - loc_2)**2) + 0.075
        loss_3 = np.sum((vector - loc_3)**2) + 0.075
        loss_5 = 3000. * np.exp( - np.sum((vector - loc_5)**2) / 0.25)
        loss_6 = 3000. * np.exp( - np.sum((vector - loc_6)**2) / 0.25)
        result = loss_0 * loss_1 * loss_2 * loss_3 + loss_5 + loss_6
        return result

    def evaluate(self, sample):
        # map sample onto hypercube
        vector = np.zeros(self.num_dims)
        for index, element in enumerate(sample):
            vector[index] = 6 * ( element / float(self.num_opts - 1) ) - 3
        return self.camel(vector)

    def is_feasible(self, params):
        x0 = params['x0']  # str
        x1 = params['x1']  # str
        sample_tuple = (x0, x1)
        if sample_tuple in self.infeas_tuples:
            return False
        return True

    @property
    def best(self):
        return ('x_13', 'x_9')
