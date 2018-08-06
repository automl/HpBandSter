import os
import unittest

import numpy as np

import statsmodels.api as sm

import ConfigSpace as CS
from hpbandster.optimizers.kde.mvkde import MultivariateKDE


from pdb import set_trace


rapid_development=True
rapid_development=False


class Base1dTest(object):
	n_train = 128
	n_test = 1024
	def setUp(self):
		self.configspace = CS.ConfigurationSpace(42)
		
		self.add_hyperparameters()

		x_train_confs = [ self.configspace.sample_configuration() for i in range(self.n_train)]
		self.x_train = np.array(	[c.get_array() for c in x_train_confs])	

		x_test_confs = [ self.configspace.sample_configuration() for i in range(self.n_test)]
		self.x_test= np.array(	[c.get_array() for c in x_test_confs])	
		
		self.sm_x_train = self.sm_transform_data(self.x_train)
		self.sm_x_test = self.sm_transform_data(self.x_test)
	
		self.sm_kde = sm.nonparametric.KDEMultivariate(data=self.sm_x_train,  var_type=self.var_types, bw='cv_ml')
		self.hp_kde_full = MultivariateKDE(self.configspace, fully_dimensional=True, fix_boundary=False)
		self.hp_kde_factor = MultivariateKDE(self.configspace, fully_dimensional=False, fix_boundary=False)
		self.hp_kde_full.fit(self.x_train,  bw_estimator='mlcv')
		self.hp_kde_factor.fit(self.x_train,  bw_estimator='mlcv')

	def sm_transform_data(self, data):
		return(data)
	
	def tearDown(self):
		self.configspace = None
		self.x_train = None
		self.x_test = None
		self.sm_kde = None
		self.hp_kde_full = None
		self.hp_kde_factor = None

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_bandwidths_estimation(self):
		# This test sometimes fails, as statsmodels uses a different optimizer with a larger tolerance
		self.assertAlmostEqual(self.sm_kde.bw[0], self.hp_kde_full.bandwidths[0], delta=2e-3)
		self.assertAlmostEqual(self.sm_kde.bw[0], self.hp_kde_factor.bandwidths[0], delta=2e-3)

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_pdfs(self):
		for bw in np.logspace(-0.5,-0.1,5):
			self.sm_kde.bw = np.array([bw])
			self.hp_kde_full.set_bandwidths(np.array([bw]))
			self.hp_kde_factor.set_bandwidths(np.array([bw]))

			p1 = self.sm_kde.pdf(self.sm_x_test)
			p2 = self.hp_kde_full.pdf(self.x_test)
			p3 = self.hp_kde_factor.pdf(self.x_test)

			self.assertTrue(np.allclose(p1, p2))
			self.assertTrue(np.allclose(p1, p3))

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_loo_likelihood(self):
		for bw in np.logspace(-1,-0.1,5):
			self.sm_kde.bw = np.array([bw])
			self.hp_kde_full.set_bandwidths(np.array([bw]))
			self.hp_kde_factor.set_bandwidths(np.array([bw]))
			
			sm_ll = self.sm_kde.loo_likelihood(bw=np.array([bw]), func=np.log)
			hp_full_ll =  self.hp_kde_full.loo_negloglikelihood()
			hp_factor_ll =  self.hp_kde_factor.loo_negloglikelihood()
			
			n = self.x_train.shape[0]
			delta = 1e-3 * np.abs((sm_ll + hp_full_ll)/2)
			# note: statsmodels' ll is not normalized, so we have to transform our result to get the same number!
			self.assertAlmostEqual(sm_ll, n*(hp_full_ll - np.log(n-1)), delta=delta)
			self.assertAlmostEqual(sm_ll, n*(hp_factor_ll - np.log(n-1)), delta=delta)




class BaseNdTest(object):
	n_train = 128
	n_test = 512
	def setUp(self):
		self.configspace = CS.ConfigurationSpace(42)
		
		self.add_hyperparameters()

		x_train_confs = [ self.configspace.sample_configuration() for i in range(self.n_train)]
		self.x_train = np.array(	[c.get_array() for c in x_train_confs])	

		x_test_confs = [ self.configspace.sample_configuration() for i in range(self.n_test)]
		self.x_test= np.array(	[c.get_array() for c in x_test_confs])	
		
		self.sm_x_train = self.sm_transform_data(self.x_train)
		self.sm_x_test = self.sm_transform_data(self.x_test)
	
		self.sm_kde = sm.nonparametric.KDEMultivariate(data=self.sm_x_train,  var_type=self.var_types, bw='cv_ml')
		
		
		self.sm_1d_kdes = [sm.nonparametric.KDEMultivariate(data=self.sm_x_train[:,i],  var_type=self.var_types[i], bw='cv_ml') for i in range(len(self.var_types))]
		
		
		self.hp_kde_full = MultivariateKDE(self.configspace, fully_dimensional=True, fix_boundary=False)
		self.hp_kde_factor = MultivariateKDE(self.configspace, fully_dimensional=False, fix_boundary=False)
		self.hp_kde_full.fit(self.x_train,  bw_estimator='mlcv')
		self.hp_kde_factor.fit(self.x_train,  bw_estimator='mlcv')

	def sm_transform_data(self, data):
		return(data)
	
	def tearDown(self):
		self.configspace = None
		self.x_train = None
		self.x_test = None
		self.sm_kde = None
		self.sm_1d_kdes = None
		self.hp_kde_full = None
		self.hp_kde_factor = None

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_bandwidths_estimation(self):
		# This test sometimes fails, as statsmodels uses a different optimizer with a larger tolerance
		
		for d in range(len(self.var_types)):
			self.assertAlmostEqual(self.sm_kde.bw[d], self.hp_kde_full.bandwidths[d], delta=5e-2)
			self.assertAlmostEqual(self.sm_1d_kdes[d].bw[0], self.hp_kde_factor.bandwidths[d], delta=5e-2)


	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_pdfs(self):
		for bw in np.logspace(-0.5,-0.1,5):
			self.sm_kde.bw = np.array([bw]*len(self.var_types))
			self.hp_kde_full.set_bandwidths(np.array([bw]*len(self.var_types)))
			self.hp_kde_factor.set_bandwidths(np.array([bw]*len(self.var_types)))

			p1 = self.sm_kde.pdf(self.sm_x_test)
			p2 = self.hp_kde_full.pdf(self.x_test)
			p3 = self.hp_kde_factor.pdf(self.x_test)
			
			p4_tmp = []
			for i, kde in enumerate(self.sm_1d_kdes):
				kde.bw = np.array([bw])
				p4_tmp.append(kde.pdf(self.sm_x_test[:,i]))

			
			p4_tmp = np.array(p4_tmp)
			p4 = np.array(p4_tmp).prod(axis=0)
			
			self.assertTrue(np.allclose(p1, p2))
			self.assertTrue(np.allclose(p3, p4))

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_loo_likelihood(self):
		for bw in np.logspace(-1,-0.1,5):
			self.sm_kde.bw = np.array([bw]*len(self.var_types))
			self.hp_kde_full.set_bandwidths(np.array([bw]*len(self.var_types)))
			self.hp_kde_factor.set_bandwidths(np.array([bw]*len(self.var_types)))
			
			sm_full_ll = self.sm_kde.loo_likelihood(bw=np.array([bw]*len(self.var_types)), func=np.log)
			hp_full_ll =  self.hp_kde_full.loo_negloglikelihood()
			hp_factor_ll =  self.hp_kde_factor.loo_negloglikelihood()

			sm_factor_ll = []
			for i, kde in enumerate(self.sm_1d_kdes):
				kde.bw = np.array([bw])
				sm_factor_ll.append(kde.loo_likelihood(bw=np.array([bw]), func=np.log))

			
			sm_factor_ll = np.array(sm_factor_ll)
			n = self.x_train.shape[0]
			delta = 1e-2 * np.abs((sm_full_ll + hp_full_ll)/2)
			# note: statsmodels' ll is not normalized, so we have to transform our result to get the same number!
			self.assertAlmostEqual(sm_full_ll, n*(hp_full_ll - np.log(n-1)), delta=delta)
			# same here, but it is easier to apply the normalization to the SM KDE's likelihoods
			delta = 1e-2 * np.abs(hp_factor_ll)
			self.assertAlmostEqual(np.sum((sm_factor_ll/n) + np.log(n-1)), hp_factor_ll , delta=delta)



class Test1dConntinuous(Base1dTest, unittest.TestCase):
	var_types='c'
	def add_hyperparameters(self):
		HPs=[]
		HPs.append( CS.UniformFloatHyperparameter('cont1', lower=0, upper=1))
		self.configspace.add_hyperparameters(HPs)

class Test1dCategorical(Base1dTest, unittest.TestCase):
	var_types='u'
	def add_hyperparameters(self):
		HPs=[]
		HPs.append( CS.CategoricalHyperparameter('cat1', choices=['foo', 'bar', 'baz']))
		self.configspace.add_hyperparameters(HPs)

class Test1dOrdinal(Base1dTest, unittest.TestCase):
	var_types='o'
	def add_hyperparameters(self):
		HPs=[]
		HPs.append( CS.OrdinalHyperparameter('ord1', ['cold', 'mild', 'warm', 'hot']))
		self.configspace.add_hyperparameters(HPs)

class Test1dInteger(Base1dTest, unittest.TestCase):
	var_types='o'
	def add_hyperparameters(self):
		HP = CS.UniformIntegerHyperparameter('int1', lower=-2, upper=2)
		self.configspace.add_hyperparameter(HP)
	def sm_transform_data(self, data):
		return(5*data - 0.5)




class Test3dConntinuous(BaseNdTest, unittest.TestCase):
	var_types='ccc'
	def add_hyperparameters(self):
		HPs=[]
		HPs.append( CS.UniformFloatHyperparameter('cont1', lower=0, upper=1))
		HPs.append( CS.UniformFloatHyperparameter('cont2', lower=0, upper=1))
		HPs.append( CS.UniformFloatHyperparameter('cont3', lower=0, upper=1))
		self.configspace.add_hyperparameters(HPs)

class Test3dMixed1(BaseNdTest, unittest.TestCase):
	var_types='uco'
	def add_hyperparameters(self):
		HPs=[]
		HPs.append( CS.CategoricalHyperparameter('cat1', choices=['foo', 'bar', 'baz']))
		HPs.append( CS.UniformFloatHyperparameter('cont1', lower=0, upper=1))
		HPs.append( CS.OrdinalHyperparameter('ord1', ['cold', 'mild', 'warm', 'hot']))
		self.configspace.add_hyperparameters(HPs)

class Test3dMixed2(BaseNdTest, unittest.TestCase):
	var_types='ucoo'
	def add_hyperparameters(self):
		HPs=[]
		HPs.append( CS.CategoricalHyperparameter('cat1', choices=['foo', 'bar', 'baz']))
		HPs.append( CS.UniformFloatHyperparameter('cont1', lower=0, upper=1))
		HPs.append( CS.UniformIntegerHyperparameter('int1', lower=-2, upper=2))
		HPs.append( CS.OrdinalHyperparameter('ord1', ['cold', 'mild', 'warm', 'hot']))
		self.configspace.add_hyperparameters(HPs)

	def sm_transform_data(self, data):
		tmp = np.copy(data)
		tmp[:,2] = 5*tmp[:,2] - 0.5
		return(tmp)



if __name__ == '__main__':
	unittest.main()
