import time
import numpy

from hpbandster.core.worker import Worker

from hpolib.benchmarks.rl.cartpole import CartpoleReduced as surrogate
from .base_worker import BaseWorker


class CartpoleReducedWorker(BaseWorker):
	def __init__(self, max_budget=None, **kwargs):

		b = surrogate()
		
		if max_budget is None:
			max_budget = b.max_budget

		super().__init__(benchmark=b, max_budget=max_budget, budget_preprocessor=int, **kwargs)

	def compute(self, config, **kwargs):
		config["batch_size"] = int(config["batch_size"])
		config["n_units_1"] = int(config["n_units_1"])
		config["n_units_2"] = int(config["n_units_2"])
		
		return(super().compute(config=config, **kwargs))

	def tpe_configspace(self):
		
		import numpy as np
		from hyperopt import hp
		
		space = {
			'learning_rate': hp.loguniform('learning_rate', np.log(1e-7), np.log(1e-1)),
			'batch_size': hp.qloguniform('batch_size', np.log(8), np.log(256), 1),
			'n_units_1': hp.qloguniform('n_units_1', np.log(8), np.log(128), 1),
			'n_units_2': hp.qloguniform('n_units_2', np.log(8), np.log(128), 1),
			'discount': hp.uniform('discount', 0, 1),
			'likelihood_ratio_clipping': hp.uniform('likelihood_ratio_clipping', 0, 1),
			'entropy_regularization': hp.uniform('entropy_regularization', 0, 1)
		}
		return(space)

