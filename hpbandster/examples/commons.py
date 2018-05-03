import time
import numpy as np
import random

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)


class MyWorker(Worker):


	def compute(self, config, budget, *args, **kwargs):
		"""
			Simple example for a compute function
			
			The loss is just a the config + some noise (that decreases with the budget)
			There is a 10 percent failure probability for any run, just to demonstrate
			the robustness of Hyperband agains these kinds of failures.
			
			For dramatization, the function sleeps for one second, which emphasizes
			the speed ups achievable with parallel workers.
		"""

		#time.sleep(1)

		# simulate some random failure
		if random.random() < 0.:
			raise RuntimeError("Random runtime error!")
		
		
		res = []
		for i in range(int(budget)):
			tmp = np.clip(config['x'] + np.random.randn()/budget, config['x']/2, 1.5*config['x'])
			res.append(tmp)
		
		return({
					'loss': np.abs(np.mean(res)),   # this is the a mandatory field to run hyperband
					'info': res             # can be used for any user-defined information - also mandatory
				})
