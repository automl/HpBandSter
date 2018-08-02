from hpolib.benchmarks.ml.bnn_benchmark import BNNOnToyFunction, BNNOnBostonHousing, BNNOnProteinStructure, BNNOnYearPrediction

from .base_worker import BaseWorker

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


class BNNWorker(BaseWorker):
	def __init__(self, dataset, path=None, **kwargs):

		if dataset == 'toyfunction':
			b = BNNOnToyFunction()
		elif dataset == 'bostonhousing':
			b = BNNOnBostonHousing()
		elif dataset == 'proteinstructure':
			b = BNNOnProteinStructure()
		elif dataset == 'yearprediction':
			b = BNNOnYearPrediction()
		else:
			raise ValueError('Unknown dataset %s!'%dataset)
		#cs = b.get_configuration_space()		
		#super().__init__(benchmark=b, configspace=cs, **kwargs)

		super().__init__(benchmark=b, **kwargs)


	def compute(self, config, **kwargs):
		config["n_units_1"] = int(config["n_units_1"])
		config["n_units_2"] = int(config["n_units_2"])
		return(super().compute(config, **kwargs))



	def tpe_configspace(self):
		from hyperopt import hp
		import numpy as np
		space = {
			'l_rate': hp.loguniform('l_rate', np.log(1e-6), np.log(1e-1)),
			'burn_in': hp.uniform('burn_in', 0, .8),
			'n_units_1': hp.qloguniform('n_units_1', np.log(16), np.log(512), 1),
			'n_units_2': hp.qloguniform('n_units_2', np.log(16), np.log(512), 1),
			'mdecay': hp.uniform('mdecay', 0, 1)
		}
		return(space)
