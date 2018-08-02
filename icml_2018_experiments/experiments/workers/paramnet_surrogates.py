import time
import numpy


from hpolib.benchmarks.surrogates.paramnet import SurrogateReducedParamNetTime as surrogate
from .base_worker import BaseWorker

class ParamNetSurrogateWorker(BaseWorker):
	budgets = { # (min, max)-budget for the different data sets
		'adult'      : (9, 243),
		'higgs'      : (9, 243),
		'letter'     : (3, 81),
		'mnist'      : (9, 243),
		'optdigits'  : (1, 27),
		'poker'      : (81, 2187),
		}

	def __init__(self, dataset, surrogate_path,*args, sleep=False, **kwargs):

		b = surrogate(dataset=dataset,path=surrogate_path)
		cs = surrogate.get_configuration_space()		
		super().__init__(benchmark=b, configspace=cs, max_budget=self.budgets[dataset][1], **kwargs)
		self.sleep = sleep


	def compute(self, config, budget, **kwargs):

		x = numpy.array([ config["x0"], config["x1"], config["x2"],
						  config["x3"],	config["x4"], config["x5"]])
		if self.sleep:	time.sleep(budget)

		return({	
					'loss': self.benchmark(x, budget=budget),
					'info': config
				})


	def tpe_configspace(self):
		
		from hyperopt import hp
		
		space = {
			'x0':  hp.uniform("x0", -6., -2.),
			'x1':  hp.uniform("x1",  3.,  8.),
			'x2':  hp.uniform("x2",  4.,  8.),
			'x3':  hp.uniform("x3", -4.,  0.),
			'x4':  hp.uniform("x4",  1.,  5.),
			'x5':  hp.uniform("x5",  0.,  .5),
		}
		return(space)

