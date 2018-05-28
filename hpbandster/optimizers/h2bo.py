import os
import time
import math
import copy
import logging

import numpy as np


import ConfigSpace as CS

from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving
from hpbandster.optimizers.config_generators.h2bo import H2BO as CG_H2BO

class H2BO(Master):
	def __init__(self,
					configspace = None,
					eta=3, min_budget=0.01, max_budget=1,
					min_points_in_model = None,	top_n_percent=15,
					num_samples = 32, random_fraction=1/3, bandwidth_factor=1,
					min_bandwidth=1e-3,bw_estimator='scott',fully_dimensional=True,
					**kwargs
					):
		"""

		Parameters
		----------
		configspace: ConfigSpace object
			valid representation of the search space
		eta : float
			In each iteration, a complete run of sequential halving is executed. In it,
			after evaluating each configuration on the same subset size, only a fraction of
			1/eta of them 'advances' to the next round.
			Must be greater or equal to 2.
		min_budget : float
			The smallest budget to consider. Needs to be positive!
		max_budget : float
			the largest budget to consider. Needs to be larger than min_budget!
			The budgets will be geometrically distributed $\sim \eta^k$ for
			$k\in [0, 1, ... , num_subsets - 1]$.
		min_points_in_model: int
			number of observations to start building a KDE. Default 'None' means
			dim+1, the bare minimum.
		top_n_percent: int
			percentage ( between 1 and 99, default 15) of the observations that are considered good.
		num_samples: int
			number of samples to optimize EI (default 64)
		random_fraction: float
			fraction of purely random configurations that are sampled from the
			prior without the model.
		bw_estimator: str
			controls the way the bandwidths are estimator. For 'scott' a quick rule of thumb based
			on the empirical variance is used, for 'mlvc' the likelihood based on 
			leave on out cross validation is maximized.
		min_bandwidth: float
			to keep diversity, even when all (good) samples have the same value for one of the parameters,
			a minimum bandwidth (Default: 1e-3) is used instead of zero. 
		iteration_kwargs: dict
			kwargs to be added to the instantiation of each iteration
		fully_dimensional: bool
			if true, the KDE is uses factored kernel across all dimensions, otherwise the PDF is a product of 1d PDFs
		"""



		# TODO: Propper check for ConfigSpace object!
		if configspace is None:
			raise ValueError("You have to provide a valid CofigSpace object")



		cg = CG_H2BO( configspace = configspace,
					min_points_in_model = min_points_in_model,
					top_n_percent=top_n_percent,
					num_samples = num_samples,
					random_fraction=random_fraction,
					bw_estimator=bw_estimator,
					min_bandwidth = min_bandwidth,
					fully_dimensional=fully_dimensional
					)

		super().__init__(config_generator=cg, **kwargs)

		# Hyperband related stuff
		self.eta = eta
		self.min_budget = min_budget
		self.max_budget = max_budget
		
		
		# precompute some HB stuff
		self.max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
		self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter-1, 0, self.max_SH_iter))



		# max total budget for one iteration
		self.budget_per_iteration = sum([b*self.eta**i for i, b in enumerate(self.budgets[::-1])])
		
		self.config.update({
						'eta'        : eta,
						'min_budget' : min_budget,
						'max_budget' : max_budget,
						'budgets'    : self.budgets,
						'max_SH_iter': self.max_SH_iter,
						'min_points_in_model' : min_points_in_model,
						'top_n_percent' : top_n_percent,
						'num_samples' : num_samples,
						'random_fraction' : random_fraction,
						'min_bandwidth': min_bandwidth,
						'bw_estimator': bw_estimator,
						'fully_dimensional': fully_dimensional,
					})

	def get_next_iteration(self, iteration, iteration_kwargs={}):
		"""
			BO-HB uses (just like Hyperband) SuccessiveHalving for each iteration.
			See Li et al. (2016) for reference.
			
			Parameters:
			-----------
				iteration: int
					the index of the iteration to be instantiated

			Returns:
			--------
				SuccessiveHalving: the SuccessiveHalving iteration with the
					corresponding number of configurations
		"""
		
		min_budget = max( self.min_budget, self.config_generator.largest_budget_with_model())
		max_budget = self.max_budget
		eta = self.eta

		# precompute some HB stuff
		max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
		budgets = max_budget * np.power(eta, -np.linspace(max_SH_iter-1, 0, max_SH_iter))

		
		# number of 'SH rungs'
		s = max_SH_iter - 1
		# number of configurations in that bracket
		n0 = int(np.floor((self.max_SH_iter)/(s+1)) * eta**s)
		ns = np.array([max(int(n0*(eta**(-i))), 1) for i in range(s+1)])
		
		while (ns * budgets[-s-1:]).sum() <= self.budget_per_iteration:
		    n0 += 1
		    ns = np.array([max(int(n0*(eta**(-i))), 1) for i in range(s+1)])
		
		n0 -= 1
		ns = np.array([max(int(n0*(eta**(-i))), 1) for i in range(s+1)])
		
		assert (ns * budgets[-s-1:]).sum() <= self.budget_per_iteration, 'Sampled iteration exceeds the budget per iteration!'
		
		return(SuccessiveHalving(HPB_iter=iteration, num_configs=ns, budgets=budgets, config_sampler=self.config_generator.get_config, **iteration_kwargs))
