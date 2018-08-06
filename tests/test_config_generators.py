import os
import unittest

import numpy as np

import ConfigSpace as CS
from hpbandster.optimizers.config_generators.bohb import BOHB
from hpbandster.core.dispatcher import Job


class TestBinaryRssRegressionForest(unittest.TestCase):

	def setUp(self):
		self.configspace = CS.ConfigurationSpace()

		self.HPs = []
		
		self.HPs.append( CS.CategoricalHyperparameter('parent', [1,2,3]))
		
		self.HPs.append( CS.CategoricalHyperparameter('child1_x1', ['foo','bar']))
		self.HPs.append( CS.UniformFloatHyperparameter('child2_x1', lower=-1, upper=1))
		self.HPs.append( CS.UniformIntegerHyperparameter('child3_x1', lower=-2, upper=5))

		self.configspace.add_hyperparameters(self.HPs)
		
		self.conditions = []
		
		self.conditions += [CS.EqualsCondition(self.HPs[1], self.HPs[0], 1)]
		self.conditions += [CS.EqualsCondition(self.HPs[2], self.HPs[0], 2)] 
		self.conditions += [CS.EqualsCondition(self.HPs[3], self.HPs[0], 3)]
		[self.configspace.add_condition(cond) for cond in self.conditions]
	
	def tearDown(self):
		self.configspace = None
		self.conditions  = None

	def test_imputation_conditional_spaces(self):
		
		bohb = BOHB(self.configspace, random_fraction=0)

		raw_array = []
		
		for i in range(128):
			
			config = self.configspace.sample_configuration()
			raw_array.append(config.get_array())
			imputed_array = bohb.impute_conditional_data(np.array(raw_array))
			self.assertFalse(np.any(np.isnan(imputed_array)))
			job = Job(i, budget=1, config = config)
			job.result = {'loss' : np.random.rand(), 'info':{}}
			bohb.new_result(job)
			
		
		for j in range(64):
			conf, info = bohb.get_config(1)
			self.assertTrue(info['model_based_pick'])

if __name__ == '__main__':
	unittest.main()
