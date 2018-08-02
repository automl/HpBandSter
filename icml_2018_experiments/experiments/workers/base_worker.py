import os
import pickle, json
import time

from hpbandster.workers.hpolibbenchmark import HPOlib2Worker
from hpbandster.core.base_iteration import Datum
from hpbandster.core.result import Result


class BaseWorker(HPOlib2Worker):

	def __init__(self, max_budget, **kwargs):
		super().__init__(**kwargs)
		
		self.time_ref = time.time()
		self.max_budget=max_budget
		self.run_data = {}

		
	def tpe_configspace(self):
		"""
			specifies the configuration space for TPE
		"""
		raise NotImplementedError("Overwrite for actual experiment")


	def subdir(self):
		"""
			specifies the subdirectory to store the data
		"""

	def evaluate_and_log (self, config, budget):
		"""
			Helper functions to log results and store them in the same format as Hyperband and BOHB runs
		"""
		
		start = time.time()
		res = self.compute(config, budget=budget)
		end = time.time()
		
		
		id = (len(self.run_data), 0,0)

		# construct a Datum object to mimic the internals of a HpBandSter iteration
		res_dict = {budget: {'loss': res['loss'], 'info': res['info']}}
		ts_dict  = {budget: {'submitted': start, 'started': start, 'finished': end}}
		self.run_data[id] = Datum(config, {}, results=res_dict, budget=budget, time_stamps = ts_dict, status='FINISHED')
		
		return(res["loss"])


	def get_result(self):
		
		# mock minial HB_config to have meaningful output
		mock_HB_config = {'min_budget': self.max_budget, 'max_budget': self.max_budget, 'time_ref': self.time_ref}
		
		# get Result by pretending to be a HB-run with one iteration
		res = Result([self.run_data, ], mock_HB_config)
		
		return(res)

	def run_tpe(self, num_iterations):
		"""
			Wrapper around TPE to return a HpBandSter Result object to integrate better with the other methods
		"""
		try:
			from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
		except ImportError:
			raise ImportError('To run TPE, please install the hyperopt package!')
		except:
			raise

		def tpe_objective(config):
			loss = self.evaluate_and_log(config, budget=self.max_budget)
			return({	'config': config,
						'loss': loss,
						'status': STATUS_OK})
						



		space = self.tpe_configspace()
		trials = Trials()
		best = fmin(tpe_objective,
				space=space,
				algo=tpe.suggest,
				max_evals=num_iterations,
				trials=trials)
		return(self.get_result())
				


	def run_smac(self, num_iterations, deterministic=True, working_directory='/tmp'):
		"""
			Wrapper around SMAC to return a HpBandSter Result object to integrate better with the other methods
		"""

		try:
			from smac.facade.smac_facade import SMAC
			from smac.scenario.scenario import Scenario
		except ImportError:
			raise ImportError('To run SMAC, please install the latest python implementation of SMAC (pip install smac)!')
		except:
			raise
		
		os.makedirs(working_directory, exist_ok=True)
		tmp_fn = os.path.join(working_directory, 'smac_%s.json'%self.run_id)
		
		# SMAC puts every call into a subprocess, so the data has to be stored on disk to
		# be persistent. Here, we simply pickle the run data to disk after every call and
		# load it before the actual call.
		with open(tmp_fn, 'wb') as fh:
			pickle.dump(self.run_data, fh)


		def smac_objective(config, **kwargs):

			with open(tmp_fn, 'rb') as fh:
				self.run_data = pickle.load(fh)
			
			loss = self.evaluate_and_log(config, budget=self.max_budget)

			with open(tmp_fn, 'wb') as fh:
				pickle.dump(self.run_data, fh)

			return loss, []
	
		scenario = Scenario({	"run_obj": "quality",
								"runcount-limit": num_iterations,
								"cs": self.configspace,
								"deterministic": deterministic,
								"initial_incumbent": "RANDOM",
								"output_dir": ""})
		

		smac = SMAC(scenario=scenario, tae_runner=smac_objective)
		smac.optimize()
		

		with open(tmp_fn, 'rb') as fh:
			self.run_data = pickle.load(fh)

		os.remove(tmp_fn)

		return(self.get_result())
