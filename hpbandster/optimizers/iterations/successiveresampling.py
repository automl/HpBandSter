import sys

from hpbandster.iterations.base import BaseIteration

import numpy as np


class SuccessiveResampling(BaseIteration):
	
	def __init__(self, *args, resampling_rate = 0.5, min_samples_advance = 1, **kwargs):
		"""
			Iteration class to resample new configurations along side keeping the good ones
			in SuccessiveHalving.
			
			Parameters:
			-----------
				resampling_rate: float
					fraction of configurations that are resampled at each stage
				min_samples_advance:int
					number of samples that are guaranteed to proceed to the next
					stage regardless of the fraction.
		
		"""
		self.resampling_rate = resampling_rate
		self.min_samples_advance = min_samples_advance


	def _advance_to_next_stage(self, config_ids, losses):
		"""
			SuccessiveHalving simply continues the best based on the current loss.
		"""
		
		ranks = np.argsort(np.argsort(losses))
		return(ranks < max(self.min_samples_advance, self.num_configs[self.stage] * (1-self.resampling_rate)) )
