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
				 min_bandwidth=1e-3, bw_estimator='scott', fully_dimensional=True,
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
			bw_estimator: string
				how the bandwidths is estimated. Possible values are 'scott' and 'mlcv' for maximum likelihood estimation
			min_bandwidth: float
				to keep diversity, even when all (good) samples have the same value for one of the parameters,
				a minimum bandwidth (Default: 1e-3) is used instead of zero. 
			fully_dimensional: bool
				if true, the KDE is uses factored kernel across all dimensions, otherwise the PDF is a product of 1d PDFs

		"""
		super().__init__(**kwargs)
		self.top_n_percent=top_n_percent
		self.configspace = configspace
		self.bw_estimator = bw_estimator
		self.min_bandwidth = min_bandwidth
		self.fully_dimensional = fully_dimensional

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
			try:
				#import pdb; pdb.set_trace()
				samples = self.kde_models[budget]['good'].sample(self.num_samples)
				ei = self.kde_models[budget]['good'].pdf(samples)/self.kde_models[budget]['bad'].pdf(samples)
				
				best_idx = np.argmax(ei)
				best_vector = samples[best_idx]

				sample = ConfigSpace.Configuration(self.configspace, vector=best_vector)


				sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
							configuration_space=self.configspace,
							configuration=sample.get_dictionary()
							)
				info_dict['model_based_pick'] = True

			except Exception as e:
				self.logger.warning(("="*50 + "\n")*3 +\
						"Error sampling a configuration!\n"+\
						"Models for budgets: %s"%(self.kde_models.keys()) +\
						"\n here is a traceback:" +\
						traceback.format_exc())

				for b,l in self.losses.items():
					self.logger.debug("budget: {}\nlosses:{}".format(b,l))
				
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
	
			
		if len(self.configs.keys()) == 1:
			min_num_points = 6
		else:
			min_num_points = self.min_points_in_model


		# skip model building if we already have a bigger model
		if max(list(self.kde_models.keys()) + [-np.inf]) > budget:
			return

		# We want to get a numerical representation of the configuration in the original space

		conf = ConfigSpace.Configuration(self.configspace, job.kwargs["config"]).get_array().tolist()
		
			
		#import pdb; pdb.set_trace()
		
		
		if conf in self.configs[budget]:
			i = self.configs[budget].index(conf)
			self.losses[budget][i].append(loss)
			print('-'*50)
			print('ran config %s with loss %f again'%(conf, loss))
		else:
			self.configs[budget].append(conf)
			self.losses[budget].append([loss])

		
		# skip model building:
		#		a) if not enough points are available
		
		tmp = np.array([np.mean(r) for r in self.losses[budget]])
		if np.sum(np.isfinite(tmp)) < min_num_points:
			self.logger.debug("Only %i successful run(s) for budget %f available, need more than %s -> can't build model!"%(np.sum(np.isfinite(tmp)), budget, min_num_points))
			return

		#		b) during warnm starting when we feed previous results in and only update once
		if not update_model:
			return



		if budget not in self.kde_models.keys():
			self.kde_models[budget] = {
				'good': MultivariateKDE(self.configspace, min_bandwidth=self.min_bandwidth, fully_dimensional=self.fully_dimensional),
				'bad' : MultivariateKDE(self.configspace, min_bandwidth=self.min_bandwidth, fully_dimensional=self.fully_dimensional)
			}	


		#import pdb; pdb.set_trace()
		num_configs = len(self.losses[budget])
		
		train_configs = np.array(self.configs[budget][-num_configs:])
		train_losses =  np.array(list(map(np.mean, self.losses[budget][-num_configs:])))

		n_good= max(3,(num_configs * self.top_n_percent) // 100)
		n_bad = num_configs-n_good

		# Refit KDE for the current budget
		idx = np.argsort(train_losses)

		train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
		train_data_bad  = self.impute_conditional_data(train_configs[idx[n_good:n_good+n_bad+1]])


		self.kde_models[budget]['bad'].fit(train_data_bad, bw_estimator=self.bw_estimator)
		self.kde_models[budget]['good'].fit(train_data_good, bw_estimator=self.bw_estimator)
		
		
		if self.bw_estimator in ['mlcv'] and n_good < 3:
			self.kde_models[budget]['good'].bandwidths[:] = self.kde_models[budget]['bad'].bandwidths

		# update probs for the categorical parameters for later sampling
		self.logger.debug('done building a new model for budget %f based on %i/%i split\nBest loss for this budget:%f\n\n\n\n\n'%(budget, n_good, n_bad, np.min(train_losses)))

