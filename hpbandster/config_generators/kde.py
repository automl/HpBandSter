import ConfigSpace
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm

from hpbandster.config_generators.base import base_config_generator


class KernelDensityEstimator(base_config_generator):
	
	def __init__(self, configspace, top_n_percent=10, update_after_n_points=50,
				 min_points_in_model = None,
				 *kwargs):
		"""
			Fits for each given budget a kernel density estimator on the best N percent of the
			evaluated configurations on this budget.


			Parameters:
			-----------
			configspace: ConfigSpace
				Configuration space object
			top_n_percent: int
				Determines the percentile of configurations that will be used as training data
				for the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered
				for training.
			update_after_n_points: int
				Specifies after how many new observed points the kernel density will be retrained.
			min_points_in_model: int
				minimum number of datapoints needed to fit a model

		"""
		super(KernelDensityEstimator, self).__init__(**kwargs)

		self.top_n_percent = top_n_percent
		self.update_after_n_points = update_after_n_points
		self.configspace = configspace
		
		self.min_points_in_model = min_points_in_model
		if min_points_in_model is None:
			self.min_points_in_model = len(self.configspace.get_hyperparameters())+1


		# TODO: so far we only consider continuous configuration spaces
		self.var_type = "c" * len(self.configspace.get_hyperparameters())
		self.configs = dict()
		self.losses = dict()
		self.kde_models = dict()
		
	def get_config(self, budget):
		"""
			Function to sample a new configuration

			This function is called inside Hyperband to query a new configuration


			Parameters:
			-----------
			budget: float
				the budget for which this configuration is scheduled

			returns: config
				should return a valid configuration

		"""
		# No observations available for this budget sample from the prior
		if len(self.kde_models.keys()) == 0:
			return self.configspace.sample_configuration().get_dictionary()
		# If we haven't seen anything with this budget, we sample from the kde trained on the highest budget
		if budget not in self.kde_models.keys():
			budget = sorted(self.kde_models.keys())[-1]
		# TODO: This only works in continuous space and with gaussian kernels
		kde = self.kde_models[budget]
		idx = np.random.randint(0, len(self.kde_models[budget].data))

		vector = [sps.truncnorm.rvs(-m/bw,(1-m)/bw, loc=m, scale=bw) for m,bw in zip(self.kde_models[budget].data[idx], kde.bw)]
		
		if np.any(np.array(vector)>1) or np.any(np.array(vector)<0):
			raise RuntimeError("truncated normal sampling problems!")
		
		sample = ConfigSpace.Configuration(self.configspace, vector=vector)
		return sample.get_dictionary(), {}

	def new_result(self, job):
		"""
			function to register finished runs

			Every time a run has finished, this function should be called
			to register it with the result logger. If overwritten, make
			sure to call this method from the base class to ensure proper
			logging.


			Parameters:
			-----------
			job_id: dict
				a dictionary containing all the info about the run
			job_result: dict
				contains all the results of the job, i.e. it's a dict with
				the keys 'loss' and 'info'

		"""

		super(KernelDensityEstimator, self).new_result(job)
		budget = job.kwargs["budget"]
		if budget not in self.configs.keys():
			self.configs[budget] = []
			self.losses[budget] = []

		# We want to get a numerical representation of the configuration in the original space
		conf = ConfigSpace.Configuration(self.configspace, job.kwargs['config'])
		self.configs[budget].append(conf.get_array())
		self.losses[budget].append(job.result['result']["loss"])


		# Check if we have enough data points to fit a KDE
		if len(self.configs[budget]) % self.update_after_n_points == 0:
			train_configs, train_losses = [], []
		
			train_configs.extend(self.configs[budget])
			train_losses.extend(self.losses[budget])

			n = int(self.top_n_percent * len(train_configs) / 100.)
			
			remaining_budgets = list(self.configs.keys())
			remaining_budgets.remove(budget)
			remaining_budgets.sort(reverse=True)


			for b in remaining_budgets:
				if  n >= self.min_points_in_model: break
				train_configs.extend(self.configs[b])
				train_losses.extend(self.losses[b])
				n = int(self.top_n_percent * len(train_configs) / 100.)

			if  len(train_losses) < self.min_points_in_model:
				return
			
			n = max(self.min_points_in_model, n)
			
			# Refit KDE for the current budget
			idx = np.argsort(train_losses)

			train_data = (np.array(train_configs)[idx])[:n]
			self.kde_models[budget] = sm.nonparametric.KDEMultivariate(data=train_data,
																 var_type=self.var_type,
																 bw='cv_ls')
