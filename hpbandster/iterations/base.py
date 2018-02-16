import sys

import numpy as np
import pdb




class SuccessiveHalving(object):
	"""
		Class to handle a run of SuccessiveHalving (SH)
	"""
	def __init__(self, iter_number, num_configs, budgets, config_sampler):
		"""
			Parameters:
			-----------

			iter_number: int
				which Hyperband Iteration this run of SH corresponds to
			num_configs: list of ints
				the number of configurations in each stage of SH
			budgets: list of floats
				the budget associated with each stage
			config_sample: callable
				a function that returns a valid configuration. Its only
				argument should be the budget that this config is first
				scheduled for. This might be used to pick configurations
				that perform best after this particular budget is exhausted
				to build a better autoML system.
		"""

		self.data = {}
		self.is_finished = False
		self.HB_iter = iter_number
		self.SH_iter = 0
		self.budgets = budgets
		self.num_configs = num_configs
		self.actual_num_configs = [0]*len(num_configs)
		self.config_sampler = config_sampler
		self.num_running = 0

	def add_configuration(self, config = None, config_info={}):
		"""
			function to add a new configuration to the current SH iteration
			
			Parameters:
			-----------
			
			config : valid configuration
				The configuration to add. If None, a configuration is sampled from the
				config_sampler
		"""
		
		if config is None:
			config, config_info = self.config_sampler(self.budgets[self.SH_iter])
		
		if self.is_finished:
			raise RuntimeError("This HB iteration is finished, you can't  add more results!")

		if self.actual_num_configs[self.SH_iter] == self.num_configs[self.SH_iter]:
			raise RuntimeError("Can't add another configuration to SH_iteration %i in HB_iteration %i."%(self.SH_iter, self.HB_iter))

		config_id = (self.HB_iter, self.SH_iter, self.actual_num_configs[self.SH_iter])

		self.data[config_id] = {
									'config'     : config,
									'config_info': config_info,
									'results'    : {},
									'time_stamps': {},
									'exceptions' : {},
									'status'     : 'QUEUED',
									'budget'     : self.budgets[self.SH_iter]
								}
		self.actual_num_configs[self.SH_iter] += 1
		return(config_id)

	def register_result(self, job):
		"""
			function to register the result of a dispy job

			This function is called from HB_master, don't call this from
			your script.
		"""

		if self.is_finished:
			raise RuntimeError("This HB iteration is finished, you can't register more results!")

		config_id = job.id
		config = job.kwargs['config']
		budget = job.kwargs['budget']
		timestamps = job.timestamps
		result = job.result
		exception = job.exception
		
		
		assert self.data[config_id]['config'] == config, 'Configurations differ!'
		assert self.data[config_id]['status'] == 'RUNNING', "Configuration wasn't scheduled for a run."
		assert self.data[config_id]['budget'] == budget, 'Budgets differ (%f != %f)!'%(self.data[config_id]['budget'], budget)


		self.data[config_id]['time_stamps'][budget] = timestamps
		self.data[config_id]['results'][budget] = result

		if (not job.result is None) and np.isfinite(result['loss']):
			self.data[config_id]['status'] = 'REVIEW'
		else:
			self.data[config_id]['status'] = 'CRASHED'
			self.data[config_id]['exceptions'][budget] = {exception}

		self.num_running -= 1
		
	def get_next_run(self):
		"""
			function to return the next configuration and budget to run.

			This function is called from HB_master, don't call this from
			your script.

			It returns None if this run of SH is finished or there are
			pending jobs that need to finish to progress to the next stage.

			If there are empty slots to be filled in the current SH stage
			(which never happens in the original SH version), a new
			configuration will be sampled and scheduled to run next.
		"""

		if self.is_finished:
			return(None)
		
		for k,v in self.data.items():
			if v['status'] == 'QUEUED':
				assert v['budget'] == self.budgets[self.SH_iter], 'Configuration budget does not align with current SH iteration!'
				v['status'] = 'RUNNING'
				self.num_running += 1
				return(k, v['config'], v['budget'])

		# check if there are still slots to fill in the current SH_iteration
		if (self.actual_num_configs[self.SH_iter] < self.num_configs[self.SH_iter]):
			self.add_configuration()
			return(self.get_next_run())

			
		if self.num_running == 0:
			# at this point an SH iteration has finished
			self.process_results()
			return(self.get_next_run())

		return(None)


	def process_results(self):
		"""
			function that is called when a stage of SH has finished and
			needs to be analyzed befor further computations

			The code here implements the original SH algorithms by
			advancing the k-best (lowest loss) configurations at the current
			budget. k is defined by the num_configs list (see __init__)
			and the current SH_iter value.

			For more advanced methods like resampling after each stage,
			overload this function only.
		"""
		self.SH_iter += 1
		
		# collect all config_ids that need to be compared
		config_ids = list(filter(lambda cid: self.data[cid]['status'] == 'REVIEW', self.data.keys()))

		if (self.SH_iter >= len(self.num_configs)):
			self.cleanup()
			return


		if len(config_ids) > 0:

			budgets = [self.data[cid]['budget'] for cid in config_ids]
			if len(set(budgets)) > 1:
				raise RuntimeError('Not all configurations have the same budget!')
			budget = budgets[0]

			results = np.array([self.data[cid]['results'][budget]['loss'] for cid in config_ids])
			ranks = np.argsort(np.argsort(results))

			advance = ranks < self.num_configs[self.SH_iter]

			for i, cid in enumerate(config_ids):
				if advance[i]:
					self.data[cid]['status'] = 'QUEUED'
					self.data[cid]['budget'] = self.budgets[self.SH_iter]
					self.actual_num_configs[self.SH_iter] += 1
				else:
					self.data[cid]['status'] = 'TERMINATED'

	def cleanup(self):
		self.is_finished = True

		for k,v in self.data.items():
			assert v['status'] in ['TERMINATED', 'REVIEW', 'CRASHED'], 'Configuration has not finshed yet!'
			del v['status']
			del v['budget']




class SuccessiveResampling(SuccessiveHalving):
	
	resampling_rate = 0.5
	min_samples_advance = 1
	
	def process_results(self):
		"""
			function that is called when a stage of SH has finished and
			needs to be analyzed befor further computations

			The code here implements the original SH algorithms by
			advancing the k-best (lowest loss) configurations at the current
			budget. k is defined by the num_configs list (see __init__)
			and the current SH_iter value.

			For more advanced methods like resampling after each stage,
			overload this function only.
		"""
		self.SH_iter += 1
		
		# collect all config_ids that need to be compared
		config_ids = list(filter(lambda cid: self.data[cid]['status'] == 'REVIEW', self.data.keys()))


		if (self.SH_iter >= len(self.num_configs)):
			self.cleanup()
			return


		if len(config_ids) > 0:

			budgets = [self.data[cid]['budget'] for cid in config_ids]
			if len(set(budgets)) > 1:
				raise RuntimeError('Not all configurations have the same budget!')
			budget = budgets[0]

			results = np.array([self.data[cid]['results'][budget]['loss'] for cid in config_ids])
			ranks = np.argsort(np.argsort(results))

			advance = ranks < max(self.min_samples_advance, self.num_configs[self.SH_iter]*(1-self.resampling_rate))

			for i, cid in enumerate(config_ids):
				if advance[i]:
					self.data[cid]['status'] = 'QUEUED'
					self.data[cid]['budget'] = self.budgets[self.SH_iter]
					self.actual_num_configs[self.SH_iter] += 1
				else:
					self.data[cid]['status'] = 'TERMINATED'
