import numpy as np
import scipy.optimize as spo

import ConfigSpace as CS

class MultivariateKDE(object):
	def __init__(self, configspace, fully_dimensional=True, min_bandwidth=1e-4):
		"""
		Parameters:
		-----------
			configspace: ConfigSpace.ConfigurationSpace object
				description of the configuration space
			fully_dimensional: bool
				if True, a true multivariate KDE is build, otherwise it's approximated by
				the product of one dimensional KDEs
				
			min_bandwidth: float
				a lower limit to the bandwidths which can insure 'uncertainty'
		
		"""
		self.configspace = configspace
		self.types, self.num_values = self._get_types()
		self.min_bandwidth=min_bandwidth
		
		self.bw_bounds = []
		
		max_bw_cont=0.5
		max_bw_cat = 0.99999
		
		for t in self.types:
			if t == 'C':
				self.bw_bounds.append((min_bandwidth, max_bw_cont))
			else:
				self.bw_bounds.append((min_bandwidth, max_bw_cat))
		
		self.bw_clip = np.array([ bwb[1] for bwb in self.bw_bounds ])
		self.fully_dimensional=fully_dimensional
		
		
		# initialize bandwidth
		self.bandwidths = np.array([float('NaN')]*len(self.types))
		self.data = None
		
	
	
	def fit(self, data, bw_estimator='scott', efficient_bw_estimation=True, update_bandwidth=True):
		"""
			fits the KDE to the data by estimating the bandwidths and storing the data
		"""
		
		if self.data is None:
			efficient_bw_estimation = False
			update_bandwidth=True
		
		self.data = np.asfortranarray(data)
		
		if not update_bandwidth:
			return
		
		if not efficient_bw_estimation or bw_estimator == 'scott':
			# inspired by the the statsmodels code
			sigmas = np.std(self.data, ddof=1, axis=0)
			IQRs = np.subtract.reduce(np.percentile(self.data, [75,25], axis=0))
			self.bandwidths = 1.059 * np.minimum(sigmas, IQRs) * np.power(self.data.shape[0], -0.2)
		
			# crop bandwidths for categorical parameters
			self.bandwidths = np.clip(self.bandwidths , self.min_bandwidth, self.bw_clip)
		
		
		if bw_estimator == 'mlcv':
			# optimize bandwidths here
			res = spo.minimize(self.loo_negloglikelihood, self.bandwidths, jac=False, bounds=self.bw_bounds, options={'ftol':1e-3}, method='SLSQP')
			self.bandwidths = res.x
			self.optimizer_result = res



	def _individual_pdfs(self, x_test=None, bandwidths=None):
		if x_test is None:
			x_test = self.data
		if bandwidths is None:
			bandwidths = self.bandwidths

		# compute the distances
		distances = (x_test[:,None,:] - self.data[None,:,:])

		pdfs = np.zeros_like(distances)

		# apply the kernels
		# note: the kernel function overwrite the values in distances!
		for i, (t, n) in enumerate(zip(self.types, self.num_values)):
			
			if n == 1:
				# single value categoricals/integers/ordinals need special treatment 
				pdfs[:,:,i] = 1
				continue
			
			if t == 'C':
				# Gaussian kernel for continuous variables
				# TODO: catch cases where the distance for a numerical value is exactly zero!
				pdfs[:,:,i] = np.exp(-0.5* np.power(distances[:,:,i]/bandwidths[i], 2))/(2.5066282746310002 * bandwidths[i])
			elif t == 'U':
				# Aitchison-Aitken kernel, used for categoricals
				idx = distances[:,:,i] == 0
				pdfs[idx, i] = 1 - bandwidths[i]
				pdfs[~idx, i] = bandwidths[i]/(n-1)

			elif t == 'I':
				# Wang-Ryzin kernel for integer parameters (note scaling by n b/c the config space rescales integer parameters to be in (0, 1) )
				idx = np.abs(distances[:,:,i]) < 1/(3*n) # distances smaller than that are considered zero
				pdfs[idx, i] = (1-bandwidths[i]) 
				pdfs[~idx, i] = 0.5*(1-bandwidths[i]) * np.power(bandwidths[i], np.abs(distances[~idx, i]*n))
			else:
				raise ValueError('Unknown type %s'%t)
		return(pdfs)


	def loo_negloglikelihood(self, bandwidths=None):
		# just renaming to avoid confusion
		pdfs = self._individual_pdfs(bandwidths=bandwidths)
		
		# get indices to remove diagonal values for LOO part :)
		indices = np.diag_indices(pdfs.shape[0])

		# combine values based on fully_dimensional!
		if self.fully_dimensional:
			
			pdfs = np.prod(pdfs, axis=-1)
			pdfs[indices] = 0
			lhs = np.sum(pdfs, axis=-1)
		else:
			pdfs[indices] = 0 # we sum first so 0 is the appropriate value
			lhs = np.prod(np.sum(pdfs, axis=-2), axis=-1)
		return(-np.sum(np.log(lhs + 1e-16)))


	def pdf(self, x_test, bandwidths=None):
		"""
			Computes the probability density function at all x_test
		"""
		N,D = self.data.shape
		x_t = x_test.reshape([-1, D])
		
		pdfs = self._individual_pdfs(x_test=x_test, bandwidths=bandwidths)

		# combine values based on fully_dimensional!
		if self.fully_dimensional:
			# first the product of the individual pdfs for each point in the data across dimensions and then the average (factorized kernel)
			pdfs = np.sum(np.prod(pdfs, axis=-1), axis=-1)/N
		else:
			# first the average over the 1d pdfs and the the product over dimensions (TPE like factorization of the pdf)
			pdfs = np.prod(np.sum(pdfs, axis=-2)/N, axis=-1)

		return(pdfs)


	def sample(self, num_samples=1):
		
		sample_indices = np.random.choice(self.data.shape[0], size=num_samples)
		samples = self.data[sample_indices]
		
		for i,(t,n) in enumerate(zip(self.types, self.num_values)):
			
			if n == 1:
				continue # parameters with only one value cannot be changed
			
			if t == 'C':
				delta = np.random.normal(size=num_samples)*self.bandwidths[i]
				samples[:,i] += delta
				oob_idx = np.argwhere(np.logical_or(samples[:,i] > 1, samples[:,i] < 0)).flatten()

				while len(oob_idx) > 0:
					samples[oob_idx,i] -= delta[oob_idx]		# revert move
					delta[oob_idx] = np.random.normal(size=len(oob_idx))*self.bandwidths[i]
					samples[oob_idx,i] += delta[oob_idx]
					oob_idx = oob_idx[np.argwhere(np.logical_or(samples[oob_idx,i] > 1, samples[oob_idx,i] < 0)).flatten()]

			elif t == 'U':
				probs = self.bandwidths[i] * np.ones(n)/(n-1)
				probs[0] = 1-self.bandwidths[i]
				
				delta = np.random.choice(n, size=num_samples, p = probs)
				
				samples[:,i] = np.mod(samples[:,i] + delta, n)
			
			
			elif t == 'I':
				
				possible_steps = np.arange(-n+1,n)
				idx = (possible_steps == 0)
				ps = 0.5*(1-self.bandwidths[i]) * np.power(self.bandwidths[i], np.abs(possible_steps))
				ps[idx] = (1-self.bandwidths[i])
				ps /= ps.sum()
				
				delta = np.zeros_like(samples[:,i])
				oob_idx = np.arange(samples.shape[0])

				while len(oob_idx) > 0:
					samples[oob_idx,i] -= delta[oob_idx]		# revert move
					delta[oob_idx] = np.random.choice(possible_steps/n, size=len(oob_idx), p=ps)
					samples[oob_idx,i] += delta[oob_idx]
					oob_idx = oob_idx[np.argwhere(np.logical_or(samples[oob_idx,i] > 1-1/(3*n), samples[oob_idx,i] < 1/(3*n))).flatten()]
				
			else:
				raise ValueError('Unknown type %s'%t)
					
		return(samples)
		

	def _get_types(self):
		""" extracts the needed types from the configspace for faster retrival later
		
			type = 0 - numerical (continuous or integer) parameter
			type >=1 - categorical parameter
			
			TODO: figure out a way to properly handle ordinal parameters
		
		"""
		types = []
		num_values = []
		for hp in self.configspace.get_hyperparameters():
			#print(hp)
			if isinstance(hp, CS.CategoricalHyperparameter):
				types.append('U')
				num_values.append(len(hp.choices))
			elif isinstance(hp, CS.UniformIntegerHyperparameter):
				types.append('I')
				num_values.append((hp.upper - hp.lower + 1))
			elif isinstance(hp, CS.UniformFloatHyperparameter):
				types.append('C')
				num_values.append(np.inf)
			#TODO: Ordinals!
			else:
				raise ValueError('Unsupported Parametertype %s'%type(hp))
		return(types, num_values)
