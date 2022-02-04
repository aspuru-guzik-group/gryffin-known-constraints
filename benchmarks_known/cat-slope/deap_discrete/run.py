#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import pickle
from gryffin import Gryffin
sys.path.append("../../")

from benchmark_functions import write_categories, save_pkl_file, load_data_from_pkl_and_continue
from benchmark_functions import CatSlopeConstr as BenchmarkSurface
from deap import base, creator, tools, algorithms

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

# --------------
# DEAP Functions
# --------------
def customMutation(individual, attrs_list, indpb=0.2, continuous_scale=0.1, discrete_scale=0.1):
    """Mutation

    Parameters
    ----------
    indpb : float
        Independent probability for each attribute to be mutated.
    """

    assert len(individual) == len(attrs_list)

    for i, attr in enumerate(attrs_list):

        # determine whether we are performing a mutation
        if np.random.random() < indpb:
            vartype = attr.__name__

            if "continuous" in vartype:
                # Gaussian perturbation with scale being 0.1 of domain range
                bound_low = attr.args[0]
                bound_high = attr.args[1]
                scale = (bound_high - bound_low) * continuous_scale
                individual[i] += np.random.normal(loc=0.0, scale=scale)
                individual[i] = _project_bounds(individual[i], bound_low, bound_high)
            elif "discrete" in vartype:
                # add/substract an integer by rounding Gaussian perturbation
                # scale is 0.1 of domain range
                bound_low = attr.args[0]
                bound_high = attr.args[1]
                scale = (bound_high - bound_low) * discrete_scale
                delta = np.random.normal(loc=0.0, scale=scale)
                individual[i] += np.round(delta, decimals=0)
                individual[i] = _project_bounds(individual[i], bound_low, bound_high)
            elif "categorical" in vartype:
                # resample a random category
                individual[i] = attr()
            else:
                raise ValueError()
        else:
            continue

    return individual,


def apply_feasibility_constraint(child, parent, param_space):

        child_vector = np.array(child, dtype=object)
        child_dict = {'x0':f'x_{child_vector[0]}', 'x1':f'x_{child_vector[1]}'}
        feasible = surface.eval_constr(child_dict)
        
        # if feasible, stop, no need to project the mutant
        if feasible is True:
            return

        # If not feasible, we try project parent or child onto feasibility boundary following these rules:
        # - for continuous parameters, we do stick breaking that is like a continuous version of a binary tree search
        #   until the norm of the vector connecting parent and child is less than a chosen threshold.
        # - for discrete parameters, we do the same until the "stick" is as short as possible, i.e. the next step
        #   makes it infeasible
        # - for categorical variables, we first reset them to the parent, then after having changed continuous
        #   and discrete, we reset the child. If feasible, we keep the child's categories, if still infeasible,
        #   we keep the parent's categories.

        parent_vector = np.array(parent, dtype=object)
        new_vector = child_vector

        continuous_mask = np.array([True if p['type'] == 'continuous' else False for p in param_space])
        discrete_mask = np.array([True if p['type'] == 'discrete' else False for p in param_space])
        categorical_mask = np.array([True if p['type'] == 'categorical' else False for p in param_space])

        child_continuous = child_vector[continuous_mask]
        child_discrete = child_vector[discrete_mask]
        child_categorical = child_vector[categorical_mask]

        parent_continuous = parent_vector[continuous_mask]
        parent_discrete = parent_vector[discrete_mask]
        parent_categorical = parent_vector[categorical_mask]

        # ---------------------------------------
        # (1) assign parent's categories to child
        # ---------------------------------------
        if any(categorical_mask) is True:
            new_vector[categorical_mask] = parent_categorical
            new_dict = {'x0':f'x_{new_vector[0]}', 'x1':f'x_{new_vector[1]}'}
            # If this fixes is, update child and return
            # This is equivalent to assigning the category to the child, and then going to step 2. Because child
            # and parent are both feasible, the procedure will converge to parent == child and will return parent
            if surface.eval_constr(new_dict) is True:
                update_individual(child, new_vector)
                return

        # -----------------------------------------------------------------------
        # (2) follow stick breaking/tree search procedure for continuous/discrete
        # -----------------------------------------------------------------------
        if any(continuous_mask) is True or any(discrete_mask) is True:
            # data needed to normalize continuous values
            lowers = np.array([d['low'] for d in param_space])
            uppers = np.array([d['high'] for d in param_space])
            inv_range = 1. / (uppers - lowers)
            counter = 0
            while True:
                # update continuous
                new_continuous = np.mean(np.array([parent_continuous, child_continuous]), axis=0)
                # update discrete, note that it can happen that child_discrete reverts to parent_discrete
                # add noise so that we can converge to the parent if needed
                noisy_mean = np.mean([parent_discrete, child_discrete], axis=0) + np.random.uniform(low=-0.1, high=0.1, size=len(parent_discrete))
                new_discrete = np.round(noisy_mean.astype(np.double), 0)

                new_vector[continuous_mask] = new_continuous
                new_vector[discrete_mask] = new_discrete
                new_dict = {'x0':f'x_{new_vector[0]}', 'x1':f'x_{new_vector[1]}'} # {'x0':new_vector[0], 'x1':new_vector[1]}

                # convergence criterion is that length of stick is less than 1% in all continuous dimensions
                # for discrete variables, parent and child should be same
                # if new_vector same as parent, we're done
                if np.sum(np.abs(parent_discrete - new_vector)) < 0.1:  # check all differences are zero
                    break

                # if child is now feasible, parent becomes new_vector (we expect parent to always be feasible)
                if surface.eval_constr(new_dict) is True:
                    parent_discrete = new_vector[discrete_mask]
                # if child still infeasible, child becomes new_vector (we expect parent to be the feasible one
                else:
                    child_discrete = new_vector[discrete_mask]

                counter += 1
                if counter > 150:  # convergence above should be reached in 128 iterations max
                    raise ValueError("constrained evolution procedure ran into trouble")

        # last parent values are the feasible ones
        new_vector[continuous_mask] = parent_continuous
        new_vector[discrete_mask] = parent_discrete

        # ---------------------------------------------------------
        # (3) Try reset child's categories, otherwise keep parent's
        # ---------------------------------------------------------
        if any(categorical_mask) is True:
            new_vector[categorical_mask] = child_categorical
            new_dict = {'x0':f'x_{new_vector[0]}', 'x1':f'x_{new_vector[1]}'} #{'x0':new_vector[0], 'x1':new_vector[1]}
            if surface.eval_constr(new_dict) is True:
                update_individual(child, new_vector)
                return
            else:
                # This HAS to be feasible, otherwise there is a bug
                new_vector[categorical_mask] = parent_categorical
                update_individual(child, new_vector)
                return
        else:
            update_individual(child, new_vector)
            return


def update_individual(ind, value_vector):
    for i, v in enumerate(value_vector):
        ind[i] = v



def cxDummy(ind1, ind2):
    """Dummy crossover that does nothing. This is used when we have a single gene in the chromosomes, such that
    crossover would not change the population.
    """
    return ind1, ind2


def create_deap_toolbox(param_space):
    from deap import base

    toolbox = base.Toolbox()
    attrs_list = []

    for i, param in enumerate(param_space):
        vartype = param['type']

        if vartype in 'continuous':
            toolbox.register(f"x{i}_{vartype}", np.random.uniform, param['low'], param['high'])

        elif vartype in 'discrete':
            toolbox.register(f"x{i}_{vartype}", np.random.randint, param['low'], param['high'])

        elif vartype in 'categorical':
            toolbox.register(f"x{i}_{vartype}", np.random.choice, param['categories'])

        attr = getattr(toolbox, f"x{i}_{vartype}")
        attrs_list.append(attr)

    return toolbox, attrs_list


def _project_bounds(x, x_low, x_high):
    if x < x_low:
        return x_low
    elif x > x_high:
        return x_high
    else:
        return x


def param_vectors_to_deap_population(param_vectors):
    population = []
    for param_vector in param_vectors:
        ind = creator.Individual(param_vector)
        population.append(ind)
    return population


def build_set_of_feasible_samples(surface, n=10):
    samples = []
    domain = np.arange(surface.num_opts)
    for x_index, x in enumerate(domain):
        for y_index, y in enumerate(domain):
            x = ['x_{}'.format(x_index), 'x_{}'.format(y_index)]
            feas = surface.eval_constr({'x0': x[0], 'x1': x[1]})
            if feas is True:
                samples.append([x_index, y_index])  
    
    np.random.seed(None)    
    sele_idx = np.random.choice(len(samples), size=n, replace=False)
    return np.array(samples)[sele_idx]


def eval_merit(X):
    X_str = [f'x_{x}' for x in X]
    y = np.squeeze(surface.run([X_str]))[()] 
    return y,


# --------------------------------
# Standardized script from here on
# --------------------------------
param_space = [{'type': 'discrete', 'low':0 , 'high':surface.num_opts-1},
               {'type': 'discrete', 'low':0 , 'high':surface.num_opts-1}]

# setup GA with DEAP
creator.create("FitnessMax", base.Fitness, weights=[-1.0])  # we minimize
creator.create("Individual", list, fitness=creator.FitnessMax)

# make toolbox
toolbox, attrs_list = create_deap_toolbox(param_space)

# crossover and mutation probabilites
CXPB, MUTPB = 0.5, 0.5


for num_repeat in range(missing_repeats):

    print(f'   Repeat {len(data_all_repeats)+1}')

    stop = False

    # initialize with 10 random samples
    samples = build_set_of_feasible_samples(surface, n=20)
    X_samples = []
    y_samples = []

    toolbox.register("population", param_vectors_to_deap_population)
    toolbox.register("evaluate", eval_merit)
    # use custom mutations for continuous, discrete, and categorical variables
    toolbox.register("mutate", customMutation, attrs_list=attrs_list, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # mating type depends on how many genes we have
    if np.shape(samples)[1] == 1:
        toolbox.register("mate", cxDummy)  # i.e. no crossover
    elif np.shape(samples)[1] == 2:
        toolbox.register("mate", tools.cxUniform, indpb=0.5)  # uniform crossover
    else:
        toolbox.register("mate", tools.cxTwoPoint)  # two-point crossover

    # Initialise population
    ngen = 0
    population = toolbox.population(samples)

    # Evaluate pop fitnesses
    fitnesses = list(map(toolbox.evaluate, np.array(population)))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # add individuals to X_samples and y_samples
    for ind in population:
        X_samples.append(ind)
        y_samples.append(ind.fitness.values[0])
        # stop optimization if we found the best
        if (f'x_{int(ind[0])}', f'x_{int(ind[1])}') == surface.best:
            stop = True
            break

    # create hall of fame
    num_elites = 2  # 2 elite individual in each population (i.e. always keep best 2)
    halloffame = tools.HallOfFame(num_elites)  # hall of fame with top individuals
    halloffame.update(population)

    fits = [ind.fitness.values[0] for ind in population]
    # print header info
    print("{0:>10}{1:>10}{2:>10}{3:>10}{4:>10}{5:>10}{6:>10}".format('gen', 'neval', 'nsamples', 'avg', 'std', 'min', 'max'))
    print("{0:>10d}{1:>10}{2:>10}{3:>10.4f}{4:>10.4f}{5:>10.4f}{6:>10.4f}".format(ngen, len(population), len(X_samples), 
                                                                              np.mean(fits), np.std(fits), 
                                                                              min(fits), max(fits)))

    # keep evaluating new offprings until we exhaust budget
    while stop is False:
        ngen += 1
        
        # size of hall of fame
        hof_size = len(halloffame.items)

        # Select the next generation individuals (allow for elitism)
        offspring = toolbox.select(population, len(population) - hof_size)
    
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < CXPB:
                parent1 = list(map(toolbox.clone, child1))  # both are parents to both children, but we select one here
                parent2 = list(map(toolbox.clone, child2))
                # mate
                toolbox.mate(child1, child2)
                # apply constraints
                apply_feasibility_constraint(child1, parent1, param_space)
                apply_feasibility_constraint(child2, parent2, param_space)
                # clear fitness values
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < MUTPB:
                parent = list(map(toolbox.clone, mutant))
                # mutate
                toolbox.mutate(mutant)
                # apply constraints
                apply_feasibility_constraint(mutant, parent, param_space)
                # clear fitness values
                del mutant.fitness.values
            
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        neval = len(invalid_ind) # how many new offprings we are evaluating
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
        # append new offsprings to X_samples and y_samples
        for ind in invalid_ind:
            # append only new samples
            if ind not in X_samples:
                X_samples.append(ind)
                y_samples.append(ind.fitness.values[0])
                # stop optimization if we found the best
                if (f'x_{int(ind[0])}', f'x_{int(ind[1])}') == surface.best:
                    stop = True
                    break

        # add the best back to population
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # replace the old population by the offspring.
        population[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in population]
        print("{0:>10d}{1:>10}{2:>10}{3:>10.4f}{4:>10.4f}{5:>10.4f}{6:>10.4f}".format(ngen, neval, len(X_samples), 
                                                                                  np.mean(fits), np.std(fits), 
                                                                                  min(fits), max(fits)))

    # store run results into a DataFrame
    x0_col = [x[0] for x in X_samples]
    x1_col = [x[1] for x in X_samples]
    obj_col = y_samples
    data = pd.DataFrame({'x0':x0_col, 'x1':x1_col, 'obj':obj_col})
    data_all_repeats.append(data)
    
    # save results to disk
    save_pkl_file(data_all_repeats)

