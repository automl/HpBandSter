import os
import time
import math
import copy
import logging

import numpy as np


import ConfigSpace as CS

from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving
from hpbandster.optimizers.config_generators.random_sampling import RandomSampling as RS

class RandomSearch(Master):
	def __init__(self, configspace = None,
					eta = 3, min_budget=1, max_budget=1,
					**kwargs
					):
		"""
                Implements a random search across the search space for comparison.
                Candidates are sampled at random and run on the maximum budget.

		Parameters
		----------
		configspace: ConfigSpace object
			valid representation of the search space
		eta : float
			In each iteration, a complete run of sequential halving is executed. In it,
			after evaluating each configuration on the same subset size, only a fraction of
			1/eta of them 'advances' to the next round.
			Must be greater or equal to 2.
		budget : float
			budget for the evaluation
		"""

		# TODO: Propper check for ConfigSpace object!
		if configspace is None:
			raise ValueError("You have to provide a valid ConfigSpace object")



		cg = RS( configspace = configspace )

		super().__init__(config_generator=cg, **kwargs)

		# Hyperband related stuff
		self.eta = eta
		self.min_budget = max_budget
		self.max_budget = max_budget
		
		
		# precompute some HB stuff
		self.max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
		self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter-1, 0, self.max_SH_iter))

		# max total budget for one iteration
		self.budget_per_iteration = sum([b*self.eta**i for i, b in enumerate(self.budgets[::-1])])
		
		self.config.update({
						'eta'        : eta,
						'min_budget' : max_budget,
						'max_budget' : max_budget,
					})

	def get_next_iteration(self, iteration, iteration_kwargs={}):
		"""
		Returns a SH iteration with only evaluations on the biggest budget
		
		Parameters
		----------
			iteration: int
				the index of the iteration to be instantiated

		Returns
		-------
			SuccessiveHalving: the SuccessiveHalving iteration with the
				corresponding number of configurations
		"""
		
		
		budgets = [self.max_budget]
		ns = [self.budget_per_iteration//self.max_budget]
		
		return(SuccessiveHalving(HPB_iter=iteration, num_configs=ns, budgets=budgets, config_sampler=self.config_generator.get_config, **iteration_kwargs))
