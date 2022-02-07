#!/usr/bin/env python

#==========================================================================

import numpy as np

from gryffin import Gryffin

import olympus
from olympus.surfaces import Surface

#==========================================================================
# `global` variables

BUDGET = 24
PLOT = True
SAMPLING_SRATEGIES = np.array([-1, 1])
surface = Surface(kind='Dejong', param_dim=2)

#==========================================================================

# gryffin config
config = {
	"general": {
		"num_cpus": 4,
		"auto_desc_gen": False,
		"batches": 1,
		"sampling_strategies": 1,
		"boosted":  False,
		"caching": True,
		"random_seed": 2021,
		"acquisition_optimizer": "adam",
		"verbosity": 3
		},
	"parameters": [
		{"name": "param_0", "type": "continuous", "low": 0.0, "high": 1.0},
		{"name": "param_1", "type": "continuous", "low": 0.0, "high": 1.0},
	],
	"objectives": [
		{"name": "obj", "goal": "min"},
	]
}

# constraints
def known_constraints(param):
	''' constraint that the sum of the parameter values must
	less than 0.8
	'''
	param_0_val = param['param_0']
	param_1_val = param['param_1']
	if param_0_val + param_1_val > 1.2:
		return False
	else:
		return True

# initialize gryffin
gryffin = Gryffin(
	config_dict=config,
	known_constraints=known_constraints,
)

#==========================================================================
# plotting instructions (optional)

if PLOT:
	import matplotlib.pyplot as plt
	import seaborn as sns
	fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
	axes = axes.flatten()
	plt.ion()

#==========================================================================

observations = []
for iter in range(BUDGET):


	# alternating sampling strategies
	select_ix = iter % len(SAMPLING_SRATEGIES)
	sampling_strategy = SAMPLING_SRATEGIES[select_ix]

	# get a new sample
	samples  = gryffin.recommend(
		observations = observations, sampling_strategies=[sampling_strategy]
	)

	sample = samples[0]

	# get measurements for samples
	observation = surface.run([val for key, val in sample.items()])[0][0]
	sample['obj'] = observation

	if PLOT:
		# optional instructions just for plotting
		for ax in axes:
			ax.clear()
		if iter >=1:
			# plotting ground truth
			x_domain = np.linspace(0., 1., 60)
			y_domain = np.linspace(0., 1., 60)
			X, Y = np.meshgrid(x_domain, y_domain)
			Z    = np.zeros((x_domain.shape[0], y_domain.shape[0]))
			acq = np.zeros((x_domain.shape[0], y_domain.shape[0]))

			for x_index, x_element in enumerate(x_domain):
				for y_index, y_element in enumerate(y_domain):
					# evaluate surface
					loss_value = surface.run([x_element, y_element])[0][0]
					Z[y_index, x_index] = loss_value
					# evaluate acquisition function
					acq_value = gryffin.get_acquisition([{'param_0': x_element, 'param_1': y_element }])[sampling_strategy][0]
					acq[y_index, x_index] = acq_value



			contours = axes[0].contour(X, Y, Z, 3, colors='black')
			axes[0].clabel(contours, inline=True, fontsize=8)
			axes[0].imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='RdGy', alpha=0.5)

			# plot the constraint line
			axes[0].plot(x_domain, 1.2-y_domain, c='k', ls='--', lw=1)

			# fill between
			axes[0].fill_between(x_domain, 1.2-y_domain, 1.2-y_domain+0.8, color='k', alpha=0.4, )

			for obs_index, obs in enumerate(observations):
				if obs_index == 0:
					axes[0].plot(obs['param_0'], obs['param_1'], marker = 'o', color = '#1a1423', markersize = 7, alpha=0.8, label='Previous observations')
				else:
					axes[0].plot(obs['param_0'], obs['param_1'], marker = 'o', color = '#1a1423', markersize = 7, alpha=0.8)

			if len(observations) >= 1:
				# plot the final observation
				axes[0].plot(observations[-1]['param_0'], observations[-1]['param_1'], marker = 'D', color = '#5b2333', markersize = 8, label='Observation')

			axes[0].set_ylim(0., 1.)

			axes[0].set_title('Constrained Dejong surface')
			axes[0].set_ylabel('param_1')
			axes[0].set_xlabel('param_0')

			axes[0].legend(loc='upper right', fontsize=10)



			# plot acquisition
			contours = axes[1].contour(X, Y, acq, 3, colors='black')
			axes[1].clabel(contours, inline=True, fontsize=8)
			axes[1].imshow(acq, extent=[0, 1, 0, 1], origin='lower', cmap='RdGy', alpha=0.5)

			# plot the constraint line
			axes[1].plot(x_domain, 1.2-y_domain, c='k', ls='--', lw=1)

			# fill between
			axes[1].fill_between(x_domain, 1.2-y_domain, 1.2-y_domain+0.8, color='k', alpha=0.4, )

			for obs_index, obs in enumerate(observations):
				if obs_index == 0:
					axes[1].plot(obs['param_0'], obs['param_1'], marker = 'o', color = '#1a1423', markersize = 7, ls='', alpha=0.8, label='Previous observations')
				else:
					axes[1].plot(obs['param_0'], obs['param_1'], marker = 'o', color = '#1a1423', markersize = 7, alpha=0.8)

			if len(observations) >= 1:
				# plot the final observation
				axes[1].plot(observations[-1]['param_0'], observations[-1]['param_1'], marker = 'D', ls='', color = '#5b2333', markersize = 8, label='Observation')

			axes[1].set_ylim(0., 1.)


			axes[1].set_title('Constrained Gryffin acquistion\n' + r'$\lambda=$'+f'{sampling_strategy}')
			axes[1].set_xlabel('param_0')



			plt.pause(0.05)


	# add measurements to cache
	observations.append(sample)
