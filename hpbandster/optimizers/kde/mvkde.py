import numpy as np
import scipy.optimize as spo



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
		self.types = self._get_types()
		self.num_categoricals = (self.types>0).sum()
		
		
		
		self.bw_bounds = []
		for t in self.types:
			if t == 0:
				self.bw_bounds.append((min_bandwidth, None))
			else:
				self.bw_bounds.append((min_bandwidth, 1))
		
		self.fully_dimensional=fully_dimensional
		
		
		# initialize bandwidth
		self.bandwidths = np.array([float('NaN')]*len(self.types))
		self.data = None
		
	
	
	def fit(self, data, efficient_bw_estimation=True):
		"""
			fits the KDE to the data by estimating the bandwidths and storing the data
		"""
		
		if self.data is None:
			efficient_bw_estimation = False
		
		self.data = np.asfortranarray(data)
		
		
		
		if not efficient_bw_estimation:
			# inspired by the the statsmodels code
			sigmas = np.std(self.data, ddof=1, axis=0)
			IQRs = np.subtract.reduce(np.percentile(self.data, [75,25], axis=0))
			self.bandwidths = 1.059 * np.minimum(sigmas, IQRs) * np.power(self.data.shape[0], -0.2)
		
			# crop bandwidths for categorical parameters
			self.bandwidths[self.types>0] = np.minimum(self.bandwidths[self.types>0] , np.ones(self.num_categoricals))
		
		# optimize bandwidths here
		res = spo.minimize(self.loo_negloglikelihood, self.bandwidths, jac=False, bounds=self.bw_bounds, options={'ftol':1e-3})#, method='SLSQP')
		self.bandwidths = res.x
		self.optimizer_result = res

	def loo_negloglikelihood(self, bandwidths=None):
		if bandwidths is None:
			bandwidths=self.bandwidths

		#import pdb; pdb.set_trace()
		# compute the distances
		distances = (self.data[:,None,:] - self.data[None,:,:])/bandwidths[None,None,:]

		# apply the kernels
		# note: the kernel function overwrite the values in distances!
		for i, t in enumerate(self.types):
			if t == 0:
				# need to catch cases where the distance for a numerical value is exactly zero!
				distances[:,:,i] += 0.1*bandwidths[i]
				
				# simple Gauss kernel
				distances[:,:,i] = np.exp(-0.5* np.power(distances[:,:,i], 2))/2.5066282746310002 / bandwidths[i]
			elif t == 1:		# single value categorical
				distances[:,:,i] = 1
			elif t > 1:
				idx = distances[:,:,i] == 0
				distances[idx, i] = 1 - bandwidths[i]
				distances[~idx, i] = bandwidths[i]/(t-1)

			else:
				raise ValueError('Unknown type %s'%t)

		# just renaming to avoid confusion
		pdfs = distances
		
		# get indices to remove diagonal values for LOO part :)
		indices = np.diag_indices(distances.shape[0])

		#import pdb; pdb.set_trace()

		# combine values based on fully_dimensional!
		if self.fully_dimensional:
			
			pdfs = np.prod(pdfs, axis=-1)
			pdfs[indices] = 0
			lhs = np.sum(pdfs, axis=-1)
		else:
			pdfs[indices] = 0 # we sum first so 0 is the appropriate value
			lhs = np.prod(np.sum(pdfs, axis=-2), axis=-1)
		print(bandwidths, -np.sum(np.log(lhs + 1e-16)))
		return(-np.sum(np.log(lhs + 1e-16)))


	def pdf(self, x_test, bandwidths=None):
		"""
			Computes the probability density function at all x_test
		"""
		N,D = self.data.shape
		x_t = x_test.reshape([-1, D])
		
		if bandwidths is None:
			bandwidths=self.bandwidths
	
		# compute the distances
		distances = (self.data[None,:,:] - x_t[:,None,:])/bandwidths[None,None,:]

		
		# apply the kernels
		# note: the kernel function overwrite the values in distances!
		for i, t in enumerate(self.types):
			if t == 0:
				distances[:,:,i] = np.exp(-0.5* np.power(distances[:,:,i], 2))/2.5066282746310002 / bandwidths[i]
			elif t == 1:		# single value categorical
				distances[:,:,i] = 1
			elif t > 1:
				idx = distances[:,:,i] == 0
				distances[idx, i] = 1 - bandwidths[i]
				distances[~idx, i] = bandwidths[i]/(t-1)		
			else:
				raise ValueError('Unknown type %s'%t)
		
		# just renaming to avoid confusion
		pdfs = distances

		# combine values based on fully_dimensional!
		if self.fully_dimensional:
			pdfs = np.sum(np.prod(pdfs, axis=-1), axis=-1)/N
		else:
			pdfs = np.prod(np.sum(pdfs, axis=-2)/N, axis=-1)

		return(pdfs)


	def sample(self, num_samples=1):
		
		sample_indices = np.random.choice(self.data.shape[0], size=num_samples)
		
		samples = self.data[sample_indices]
		
		
		for i,t in enumerate(self.types):
			
			if t == 0:
				
				delta = np.random.normal(size=num_samples)*self.bandwidths[i]
				samples[:,i] += delta
				oob_idx = np.argwhere(np.logical_or(samples[:,i] > 1, samples[:,i] < 0)).flatten()

				while len(oob_idx) > 0:
					samples[oob_idx,i] -= delta[oob_idx]		# revert move
					delta[oob_idx] = np.random.normal(size=len(oob_idx))*self.bandwidths[i]
					samples[oob_idx,i] += delta[oob_idx]
					oob_idx = oob_idx[np.argwhere(np.logical_or(samples[oob_idx,i] > 1, samples[oob_idx,i] < 0)).flatten()]


			elif t == 1:	# this is a categorical parameter with only one value, so there are no choices
				continue

			elif t > 1:
				probs = self.bandwidths[i] * np.ones(t)/(t-1)
				probs[0] = 1-self.bandwidths[i]
				
				delta = np.random.choice(t, size=num_samples, p = probs)
				
				samples[:,i] = np.mod(samples[:,i] + delta, t)
			
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
		for hp in self.configspace.get_hyperparameters():
			
			if hasattr(hp, 'choices'):			# categorical parameter
				types.append(len(hp.choices))
			elif hasattr(hp, 'check_int'): 	# integer parameter
				types.append(-1)
			else:
				types.append(0)					# continuous parameter
		return(np.array(types))



if __name__ == "__main__":
	import ConfigSpace as CS
	
	cs = CS.ConfigurationSpace()
	
	cs.add_hyperparameter(CS.UniformFloatHyperparameter('foo', lower=0, upper=1))
	cs.add_hyperparameter(CS.CategoricalHyperparameter('bar', choices=['bla', 'blubs', '??']))
	
	N=256

	x_train = np.array([cs.sample_configuration().get_array() for i in range(N)])
	
	kde = MultivariateKDE(cs)
	kde.fit(x_train)
	print(kde.bandwidths)
	
	import statsmodels.api as sm
	sm_kde = sm.nonparametric.KDEMultivariate(data=x_train,  var_type='uc', bw='cv_ml')
	
	samples = kde.sample(4)
	
	print(sm_kde.bw)
	print(kde.bandwidths)
	
	print(sm_kde.pdf(samples))
	print(kde.pdf(samples))
	import  pdb; pdb.set_trace()
