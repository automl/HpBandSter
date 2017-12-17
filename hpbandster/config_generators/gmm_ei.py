import ConfigSpace
import numpy as np
import scipy.stats as sps
import scipy.optimize as spo

from sklearn import mixture


from hpbandster.config_generators.base import base_config_generator


class GMMEI(base_config_generator):
	
	def __init__(self, configspace, top_n_percent=10, update_after_n_points=16,
				 max_num_components = 5,
				 min_points_in_model = None, mode='DE',
				 num_samples = 1024,
				 **kwargs):
		"""
			Fits for each given budget a GMM on the best N percent of the
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
			max_num_components: int
				number of independent Gaussians
			min_points_in_model: int
				minimum number of datapoints needed to fit a model
			mode: str
				how EI is optimized:
					-'sampling' corresponds to sampling from the KDE and evaluating EI
					-'DE' corresponds to differential evolution (from scipy.optimize)
			num_samples: int
				number of samples drawn to optimize EI via sampling or number of function
				when using DE

		"""
		super(GMMEI, self).__init__(**kwargs)

		self.top_n_percent = top_n_percent
		self.update_after_n_points = update_after_n_points
		self.configspace = configspace
		
		
		self.max_num_components = max_num_components
		self.min_points_in_model = min_points_in_model
		if min_points_in_model is None:
			self.min_points_in_model = len(self.configspace.get_hyperparameters())+1

		self.mode=mode
		self.num_samples = num_samples


		# TODO: so far we only consider continuous configuration spaces
		self.var_type = "c" * len(self.configspace.get_hyperparameters())
		self.configs = dict()
		self.losses = dict()
		self.gmm_models = dict()
		
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
		# If no model is available, sample from prior
		if len(self.gmm_models.keys()) == 0:
			return self.configspace.sample_configuration().get_dictionary()
		
		# If we haven't seen anything with this budget, we sample from the kde trained on the highest budget
		if budget not in self.gmm_models.keys():
			budget = max(self.gmm_models.keys())
		
		
		ll = self.gmm_models[budget]['good'].score_samples
		lg = self.gmm_models[budget]['bad' ].score_samples
		
		minimize_me = lambda x: lg(x.reshape([1,-1])) - ll(x.reshape([1,-1]))
	
		if self.mode == 'DE':

			dim = len(self.configspace._hyperparameters)
			# the 15*dim is the default population size of spo.differential_evolution
			maxiter = self.num_samples//(15*dim) + 1
		
			res = spo.differential_evolution(minimize_me, [(0,1)]*dim, maxiter=maxiter, init='random')
			
			sample = ConfigSpace.Configuration(self.configspace, vector=res.x)
			
			
		if self.mode == 'sampling':
			
			raise NotImplementedError('TODO')
		
		print(sample.get_array())
		return sample.get_dictionary()

	def new_result(self, job):
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


		super(GMMEI, self).new_result(job)
		budget = job.kwargs["budget"]
		if budget not in self.configs.keys():
			self.configs[budget] = []
			self.losses[budget] = []

		# We want to get a numerical representation of the configuration in the original space
		conf = ConfigSpace.Configuration(self.configspace, job.kwargs['config'])
		self.configs[budget].append(conf.get_array())
		self.losses[budget].append(job.result["result"]["loss"])

		if len(self.configs[budget]) <= self.min_points_in_model:
			return 

		if len(self.configs[budget]) % self.update_after_n_points == 0:
			train_configs = np.array(self.configs[budget])
			train_losses =  np.array(self.losses[budget])
		
			idx = np.argsort(train_losses)
			n_good = min(self.max_num_components, self.top_n_percent * train_losses.shape[0])
			
			self.gmm_models[budget] = {
				'good': mixture.BayesianGaussianMixture(n_components=1, covariance_type='full').fit(train_configs[idx[:n_good],:]),
				'bad' : mixture.BayesianGaussianMixture(n_components=self.max_num_components, covariance_type='spherical').fit(train_configs[idx[n_good:],:])
			}
