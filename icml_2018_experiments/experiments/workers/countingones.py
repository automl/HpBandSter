from hpolib.benchmarks.synthetic_functions.counting_ones import CountingOnes

from .base_worker import BaseWorker
import time

class CountingOnesWorker(BaseWorker):
	def __init__(self, num_continuous, num_categorical, **kwargs):
		
		self.num_cont = num_continuous
		self.num_cat  = num_categorical

		b = CountingOnes()
		cs = CountingOnes.get_configuration_space(n_categorical=num_categorical, n_continuous = num_continuous)		
		
		super().__init__(benchmark=b, configspace=cs,**kwargs)
	#def compute(self, *args, **kwargs):
	#	time.sleep(0.01)
	#	return(super().compute(*args, **kwargs))
		
	def tpe_configspace(self):
		
		from hyperopt import hp
		
		space = {}

		for d in range(self.num_cont):
			space["float_%d" % d] = hp.uniform("float_%d" % d, 0, 1)

		for d in range(self.num_cat):
			space["cat_%d" % d] = hp.choice("cat_%d" % d, [0, 1])
		return(space)


