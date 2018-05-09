import os
import unittest

import numpy as np

import statsmodels.api as sm

import ConfigSpace as CS
from hpbandster.optimizers.kde.mvkde import MultivariateKDE


from pdb import set_trace


class Test1dContinuous(unittest.TestCase):
	n_train = 256
	n_test = 1024
	def setUp(self):
		self.configspace = CS.ConfigurationSpace(42)
		
		HPs=[]
		HPs.append( CS.UniformFloatHyperparameter('cont1', lower=0, upper=1))
		self.configspace.add_hyperparameters(HPs)
		
		x_train_confs = [ self.configspace.sample_configuration() for i in range(self.n_train)]
		self.x_train = np.array(	[c.get_array() for c in x_train_confs])	


		x_test_confs = [ self.configspace.sample_configuration() for i in range(self.n_test)]
		self.x_test= np.array(	[c.get_array() for c in x_train_confs])	
		
		
		self.sm_kde = sm.nonparametric.KDEMultivariate(data=self.x_train,  var_type='c', bw='cv_ml')
		self.hp_kde = MultivariateKDE(self.configspace, fully_dimensional=True)
		self.hp_kde.fit(self.x_train)
		
		
	def tearDown(self):
		self.configspace = None
		self.x_train = None
		self.x_test = None
		self.sm_kde = None
		self.hp_kde = None

	def test_bandwidths_estimation(self):
		# This test sometimes fails, as statsmodels uses a different optimizer with a larger tolerance
		self.assertAlmostEqual(self.sm_kde.bw[0], self.hp_kde.bandwidths[0], 2)

	def test_pdf(self):
		self.assertTrue(np.allclose(self.sm_kde.pdf(self.x_test), self.hp_kde.pdf(self.x_test, bandwidths=self.sm_kde.bw)))
		
		for bw in np.logspace(-2.5,0,20):
			self.sm_kde.bw = np.array([bw])
			self.assertTrue(np.allclose(self.sm_kde.pdf(self.x_test), self.hp_kde.pdf(self.x_test, bandwidths=np.array([bw]))))

	def test_loo_likelihood(self):
		self.assertTrue(np.allclose(self.sm_kde.pdf(self.x_test), self.hp_kde.pdf(self.x_test, bandwidths=self.sm_kde.bw)))

		for bw in np.logspace(-2.5,0,20):
			self.sm_kde.bw = np.array([bw])
			self.assertAlmostEqual(self.sm_kde.loo_likelihood(bw=np.array([bw]), func=lambda x:np.log(x)), self.hp_kde.loo_negloglikelihood(bandwidths=np.array([bw])))
	

class Test1dCategorical(unittest.TestCase):
	n_train = 256
	n_test = 1024
	def setUp(self):
		self.configspace = CS.ConfigurationSpace(43)
		
		HPs=[]
		HPs.append( CS.CategoricalHyperparameter('cat1', choices=['foo', 'bar', 'baz']))
		self.configspace.add_hyperparameters(HPs)
		
		x_train_confs = [ self.configspace.sample_configuration() for i in range(self.n_train)]
		self.x_train = np.array(	[c.get_array() for c in x_train_confs])	


		x_test_confs = [ self.configspace.sample_configuration() for i in range(self.n_test)]
		self.x_test= np.array(	[c.get_array() for c in x_train_confs])	
		
		
		self.sm_kde = sm.nonparametric.KDEMultivariate(data=self.x_train,  var_type='u', bw='cv_ml')
		self.hp_kde = MultivariateKDE(self.configspace, fully_dimensional=True)
		self.hp_kde.fit(self.x_train)
		
		
	def tearDown(self):
		self.configspace = None
		self.x_train = None
		self.x_test = None
		self.sm_kde = None
		self.hp_kde = None

	def test_bandwidths_estimation(self):
		# This test sometimes fails, as statsmodels uses a different optimizer with a larger tolerance
		self.assertAlmostEqual(self.sm_kde.bw[0], self.hp_kde.bandwidths[0], 2)

	def test_pdf(self):

		self.assertTrue(np.allclose(self.sm_kde.pdf(self.x_test), self.hp_kde.pdf(self.x_test, bandwidths=self.sm_kde.bw)))
		
		for bw in np.logspace(-2.5,0,20):
			self.sm_kde.bw = np.array([bw])
			self.assertTrue(np.allclose(self.sm_kde.pdf(self.x_test), self.hp_kde.pdf(self.x_test, bandwidths=np.array([bw]))))

	def test_loo_likelihood(self):

		self.assertTrue(np.allclose(self.sm_kde.pdf(self.x_test), self.hp_kde.pdf(self.x_test, bandwidths=self.sm_kde.bw)))

		for bw in np.logspace(-2.5,0,20):
			self.sm_kde.bw = np.array([bw])
			self.assertAlmostEqual(self.sm_kde.loo_likelihood(bw=np.array([bw]), func=lambda x:np.log(x)), self.hp_kde.loo_negloglikelihood(bandwidths=np.array([bw])))


class Test1dInteger(unittest.TestCase):
	n_train = 25
	n_test = 102
	def setUp(self):
		self.configspace = CS.ConfigurationSpace(43)
		
		HPs=[]
		HPs.append( CS.UniformIntegerHyperparameter('int1', lower=-2, upper=2))
		self.configspace.add_hyperparameters(HPs)
		
		x_train_confs = [ self.configspace.sample_configuration() for i in range(self.n_train)]
		self.x_train = np.array([c.get_array() for c in x_train_confs])	


		x_test_confs = [ self.configspace.sample_configuration() for i in range(self.n_test)]
		self.x_test= np.array(	[c.get_array() for c in x_test_confs])	
		
		self.sm_kde = sm.nonparametric.KDEMultivariate(data=np.rint(5*self.x_train - 0.5),  var_type='o', bw='cv_ml')
		self.hp_kde = MultivariateKDE(self.configspace, fully_dimensional=False)
		self.hp_kde.fit(self.x_train)
		
		
	def tearDown(self):
		self.configspace = None
		self.x_train = None
		self.x_test = None
		self.sm_kde = None
		self.hp_kde = None

	def test_bandwidths_estimation(self):
		# This test sometimes fails, as statsmodels uses a different optimizer with a larger tolerance
		self.assertAlmostEqual(self.sm_kde.bw[0], self.hp_kde.bandwidths[0], delta=1e-3)

	def test_pdf(self):
		self.assertTrue(np.allclose(self.sm_kde.pdf(np.rint(self.x_test*5-0.5)), self.hp_kde.pdf(self.x_test, bandwidths=self.sm_kde.bw)))
		
		for bw in np.logspace(-2.5,-0.1,20):
			self.sm_kde.bw = np.array([bw])
			self.assertTrue(np.allclose(self.sm_kde.pdf(np.rint(self.x_test*5-0.5)), self.hp_kde.pdf(self.x_test, bandwidths=np.array([bw])),5))


	def test_loo_likelihood(self):
		for bw in np.logspace(-2.5,-0.1,20):
			self.sm_kde.bw = np.array([bw])
			self.assertAlmostEqual(self.sm_kde.loo_likelihood(bw=np.array([bw]), func=np.log), self.hp_kde.loo_negloglikelihood(bandwidths=np.array([bw])), delta=1e-3)
	
	
	
	def test_sampling_1(self):
		"""
			Fit KDE on 100 symmetrically distributed points and then draw samples.
			B/c of the symmetry, the resulting frequencies have to match the data
			in the KDE. If it was asymmetric, the boundary effects would be more
			dominant and might skrew things up!
		"""
		
		ns = np.array([10, 20, 40, 20, 10])
		vals = np.array([-2, -1,0,1,2])
		
		x_train = []
		for n,v in zip(ns, vals):
			for i in range(n):
				x_train.append(CS.Configuration(self.configspace, {'int1' : v}).get_array())
		
		x_train=np.array(x_train)
		self.hp_kde.fit(x_train)



		num_samples = 2**15
		samples = self.hp_kde.sample(num_samples)
		samples = np.array([CS.Configuration(self.configspace, vector=s)['int1'] for s in samples], dtype=np.int)


		for v, n in zip(vals, ns):
			self.assertAlmostEqual((samples == v).sum()/num_samples, n/ns.sum(), delta=5e-3)
	

	def test_sampling_2(self):
		"""
			Fit KDE on just one point and compares the empirical distribution
			of the samples to the PDF. Within the standard MC error, these match
			exactly.
		"""
		vals = np.array([-2, -1,0,1,2])		
		bandwidth = 0.5

		for loc in vals[:3]:
			x_train = np.array([CS.Configuration(self.configspace, {'int1' : loc}).get_array()])
			
			x_train=np.array(x_train)
			self.hp_kde.fit(x_train)
			self.hp_kde.bandwidths = np.array([bandwidth])

			num_samples = 2**16
			samples = self.hp_kde.sample(num_samples)
			samples = np.array([CS.Configuration(self.configspace, vector=s)['int1'] for s in samples], dtype=np.int)

			# compute expected frequencies via the pdf function
			ps = self.hp_kde.pdf((np.linspace(1/(2*len(vals)),1-1/(2*len(vals)), len(vals))).reshape([-1,1]))
			ps /= ps.sum()

			p_hats = np.array([(samples == v).sum()/num_samples for v in vals])

			for p, p_hat in zip(ps, p_hats):
				self.assertAlmostEqual(p, p_hat, delta=5e-3)


		
		
		
		
if __name__ == '__main__':
	unittest.main()
