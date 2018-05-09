import logging
from copy import deepcopy
import traceback


import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps

from hpbandster.core.base_config_generator import base_config_generator
from hpbandster.optimizers.kde.mvkde import MultivariateKDE

class H2BO(base_config_generator):
	def __init__(self, configspace, min_points_in_model = None,
				 top_n_percent=15, num_samples = 64, random_fraction=1/3,
				 bandwidth_factor=3, min_bandwidth=1e-3,
				**kwargs):
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
			min_points_in_model: int
				minimum number of datapoints needed to fit a model
			num_samples: int
				number of samples drawn to optimize EI via sampling
			random_fraction: float
				fraction of random configurations returned
			bandwidth_factor: float
				widens the bandwidth for contiuous parameters for proposed points to optimize EI
			min_bandwidth: float
				to keep diversity, even when all (good) samples have the same value for one of the parameters,
				a minimum bandwidth (Default: 1e-3) is used instead of zero. 

		"""
		super().__init__(**kwargs)
		self.top_n_percent=top_n_percent
		self.configspace = configspace
		self.bw_factor = bandwidth_factor
		self.min_bandwidth = min_bandwidth

		self.min_points_in_model = min_points_in_model
		if min_points_in_model is None:
			self.min_points_in_model = len(self.configspace.get_hyperparameters())+1
		
		#if self.min_points_in_model < len(self.configspace.get_hyperparameters())+1:
		#	self.logger.warning('Invalid min_points_in_model value. Setting it to %i'%(len(self.configspace.get_hyperparameters())+1))
		#	self.min_points_in_model =len(self.configspace.get_hyperparameters())+1
		
		self.num_samples = num_samples
		self.random_fraction = random_fraction


		self.configs = dict()
		self.losses = dict()
		self.good_config_rankings = dict()
		self.kde_models = dict()


	def largest_budget_with_model(self):
		if len(self.kde_models) == 0:
			return(-np.inf)
		return(max(self.kde_models.keys()))

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
		sample = None
		info_dict = {}
		
		# If no model is available, sample from prior
		# also mix in a fraction of random configs
		if len(self.kde_models.keys()) == 0 or np.random.rand() < self.random_fraction:
			sample =  self.configspace.sample_configuration()
			info_dict['model_based_pick'] = False


		if sample is None:
			samples = self.kde_models[budget]['good'].sample(self.num_samples)
			ei = self.kde_models[budget]['good'].pdf(samples)/self.kde_models[budget]['bad'].pdf(samples)
			
			best_idx = np.argmax(ei)
			best_vector = samples[best_idx]

			sample = ConfigSpace.Configuration(self.configspace, vector=best_vector)

			try:
				sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
							configuration_space=self.configspace,
							configuration=sample.get_dictionary()
							)
				info_dict['model_based_pick'] = True

			except Exception as e:
				self.logger.warning(("="*50 + "\n")*3 +\
						"Error converting configuration:\n%s"%sample+\
						"\n here is a traceback:" +\
						traceback.format_exc())
				
				sample =  self.configspace.sample_configuration()
				info_dict['model_based_pick'] = False
				
		return sample.get_dictionary(), info_dict



	def impute_conditional_data(self, array):

		return_array = np.empty_like(array)

		for i in range(array.shape[0]):
			datum = np.copy(array[i])
			nan_indices = np.argwhere(np.isnan(datum)).flatten()

			while (np.any(nan_indices)):
				nan_idx = nan_indices[0]
				valid_indices = np.argwhere(np.isfinite(array[:,nan_idx])).flatten()

				if len(valid_indices) > 0:
					# pick one of them at random and overwrite all NaN values
					row_idx = np.random.choice(valid_indices)
					datum[nan_indices] = array[row_idx, nan_indices]

				else:
					# no good point in the data has this value activated, so fill it with a valid but random value
					t = self.vartypes[nan_idx]
					if t == 0:
						datum[nan_idx] = np.random.rand()
					else:
						datum[nan_idx] = np.random.randint(t)

				nan_indices = np.argwhere(np.isnan(datum)).flatten()
			return_array[i,:] = datum
		return(return_array)

	def new_result(self, job, update_model=True):
		"""
			function to register finished runs

			Every time a run has finished, this function should be called
			to register it with the result logger. If overwritten, make
			sure to call this method from the base class to ensure proper
			logging.


			Parameters:
			-----------
			job: hpbandster.distributed.dispatcher.Job object
				contains all the info about the run
		"""

		super().new_result(job)

		if job.result is None:
			# One could skip crashed results, but we decided 
			# assign a +inf loss and count them as bad configurations
			loss = np.inf
		else:
			loss = job.result["loss"]

		budget = job.kwargs["budget"]

		if budget not in self.configs.keys():
			self.configs[budget] = []
			self.losses[budget] = []
	
			

		# skip model building if we already have a bigger model
		if max(list(self.kde_models.keys()) + [-np.inf]) > budget:
			return

		# We want to get a numerical representation of the configuration in the original space

		conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"])
		self.configs[budget].append(conf.get_array())
		self.losses[budget].append(loss)

		
		# skip model building:
		#		a) if not enough points are available
		if len(self.configs[budget]) < self.min_points_in_model:
			self.logger.debug("Only %i run(s) for budget %f available, need more than %s -> can't build model!"%(len(self.configs[budget]), budget, self.min_points_in_model))
			return

		#		b) during warnm starting when we feed previous results in and only update once
		if not update_model:
			return



		if budget not in self.kde_models.keys():
			self.kde_models[budget] = {
				'good': MultivariateKDE(self.configspace, min_bandwidth=self.min_bandwidth),
				'bad' : MultivariateKDE(self.configspace, min_bandwidth=self.min_bandwidth)
			}	

		train_configs = np.array(self.configs[budget])
		train_losses =  np.array(self.losses[budget])

		n_good= min(32, max(1+0*self.min_points_in_model, (self.top_n_percent * train_configs.shape[0])//100 ))
		n_bad = max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100)

		# Refit KDE for the current budget
		idx = np.argsort(train_losses)

		train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
		train_data_bad  = self.impute_conditional_data(train_configs[idx[n_good:n_good+n_bad]])


		self.kde_models[budget]['bad'].fit(train_data_bad, bw_estimator='scott')
		self.kde_models[budget]['good'].fit(train_data_good, bw_estimator='scott')
		
		#if n_good < self.min_points_in_model:
		#	self.kde_models[budget]['good'].bandwidths[:] = self.kde_models[budget]['bad'].bandwidths
		
		
		"""
		print('='*50)
		print(self.kde_models[budget]['good'].bandwidths)
		print('best:\n',self.kde_models[budget]['good'].data[0])
		print(self.kde_models[budget]['good'].data.mean(axis=0))
		
		print(self.kde_models[budget]['bad'].bandwidths)
		print(self.kde_models[budget]['bad'].data.shape)
		"""
		# update probs for the categorical parameters for later sampling
		self.logger.debug('done building a new model for budget %f based on %i/%i split\nBest loss for this budget:%f\n\n\n\n\n'%(budget, n_good, n_bad, np.min(train_losses)))

