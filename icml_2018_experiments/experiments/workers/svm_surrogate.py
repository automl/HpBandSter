
import argparse
import numpy

from hpbandster.core.worker import Worker

from hpolib.benchmarks.surrogates.svm import SurrogateSVM as surrogate
from .base_worker import BaseWorker

class SVMSurrogateWorker(BaseWorker):
	def __init__(self, surrogate_path=None, sleep=False, **kwargs):

		b = surrogate(path=surrogate_path)
		cs = surrogate.get_configuration_space()
		kwargs.update({'max_budget': 1.})
		kwargs.update({'budget_name': 'dataset_fraction'})
		super().__init__(benchmark=b, configspace=cs, **kwargs)
		self.sleep = sleep


	def tpe_configspace(self):
		
		from hyperopt import hp
		
		space = {
			'x0':  hp.uniform("x0", -10.,  10.),
			'x1':  hp.uniform("x1",  -10., 10.),
		}
		return(space)
