from hpbandster.core.base_iteration import BaseIteration
import numpy as np


class SuccessiveHalving(BaseIteration):

	def _advance_to_next_stage(self, config_ids, losses):
		"""
			SuccessiveHalving simply continues the best based on the current loss.
		"""
		ranks = np.argsort(np.argsort(losses))
		return(ranks < self.num_configs[self.stage])
