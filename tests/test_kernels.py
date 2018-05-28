import os
import unittest

import numpy as np

#from scipy.integrate import quadrature as quadrature
from scipy.integrate import quad as quadrature


from statsmodels.nonparametric import kernels as sm_kernels
from hpbandster.optimizers.kde import kernels as hp_kernels



import ConfigSpace as CS

from pdb import set_trace


rapid_development=True
rapid_development=False

class TestGaussian(unittest.TestCase):
	n_train = 256
	n_test = 1024
	def setUp(self):
		self.x_train = np.random.rand(self.n_train)
		self.x_test = np.random.rand(self.n_test)
		
	def tearDown(self):
		self.x_train = None
		self.x_test = None

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_values(self):

		for bw in [1e-3, 1e-2, 1e-1, 1]:
			sm_values = sm_kernels.gaussian(bw, self.x_train[:,None],  self.x_test[None,:])
			hp_kernel = hp_kernels.Gaussian(data=self.x_train, bandwidth=bw, fix_boundary=False)
			hp_values = hp_kernel(self.x_test)
			self.assertTrue(np.allclose(hp_values, sm_values/bw, 1e-4))

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_pdf_boundary_simple(self):
		self.x_train = np.array([0])
		for bw in [1e-3, 1e-2, 1e-1]:
			# note: for larger bandwidths, the pdf also needs to be truncated as +1,
			# which leads to something different than twice the pdf
		
			hp_kernel1 = hp_kernels.Gaussian(data=self.x_train, bandwidth=bw, fix_boundary=False)
			hp_kernel2 = hp_kernels.Gaussian(data=self.x_train, bandwidth=bw, fix_boundary=True)
		
			hp_values1 = hp_kernel1(self.x_test)
			hp_values2 = hp_kernel2(self.x_test)
			self.assertTrue(np.allclose(2*hp_values1, hp_values2, 1e-4))
		
		self.x_train = np.array([1])
		for bw in [1e-3, 1e-2, 1e-1]:
			# note: for larger bandwidths, the pdf also needs to be truncated as +1,
			# which leads to something different than twice the pdf
		
			hp_kernel1 = hp_kernels.Gaussian(data=self.x_train, bandwidth=bw, fix_boundary=False)
			hp_kernel2 = hp_kernels.Gaussian(data=self.x_train, bandwidth=bw, fix_boundary=True)
		
			hp_values1 = hp_kernel1(self.x_test)
			hp_values2 = hp_kernel2(self.x_test)
			
			self.assertTrue(np.allclose(2*hp_values1, hp_values2, 1e-4))

		
		# simple test based on 68, 95, 99% rule
		self.x_train = np.array([0.5])
		for bw, w in ([0.5, 0.6827], [0.25, 0.9545], [1/6, 0.9973]):
			hp_kernel = hp_kernels.Gaussian(data=self.x_train, bandwidth=bw, fix_boundary=True)
			self.assertAlmostEqual(hp_kernel.weights[0], 1/w, delta=1e-4)

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_pdf_boundary_quadrature(self):
		for bw in [1e-2, 1e-1, 1]:
			hp_kernel = hp_kernels.Gaussian(data=self.x_train, bandwidth=bw, fix_boundary=True)
		
			def quad_me(x):
				x_test = np.array([x])
				pdfs = hp_kernel(x_test)
				return(pdfs.mean())

			self.assertAlmostEqual(quadrature(quad_me, 0, 1)[0], 1, delta=1e-4)
		

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_sample(self):
		
		num_samples = 2**20
	
		for bw in [1e-1, 5e-1, 1]:
			hp_kernel = hp_kernels.Gaussian(data=self.x_train, bandwidth=bw, fix_boundary=True)

			samples = hp_kernel.sample(num_samples=num_samples)
			phat1, x = np.histogram(samples, normed=True)
			phat2 = hp_kernel((x[1:] + x[:-1])/2).mean(axis=0)
			
			for p1, p2 in zip(phat1, phat2):
				self.assertAlmostEqual(p1, p2, delta=5e-2)

class Test1dCategorical(unittest.TestCase):
	n_train = 256
	n_test = 1024
	def setUp(self):
		self.configspace = CS.ConfigurationSpace(43)
		
		HPs=[]
		HPs.append( CS.CategoricalHyperparameter('cat1', choices=['foo', 'bar', 'baz']))
		self.configspace.add_hyperparameters(HPs)
		
		x_train_confs = [ self.configspace.sample_configuration() for i in range(self.n_train)]
		self.x_train = np.array(	[c.get_array() for c in x_train_confs]).squeeze()

		x_test_confs = [ self.configspace.sample_configuration() for i in range(self.n_test)]
		self.x_test= np.array(	[c.get_array() for c in x_train_confs]).squeeze()
		
		
	def tearDown(self):
		self.configspace = None
		self.x_train = None
		self.x_test = None

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_values(self):
		for bw in [1e-3, 1e-2, 1e-1, 1]:
			sm_values = []
			for x in self.x_test:
				sm_values.append(sm_kernels.aitchison_aitken(bw, self.x_train, x))
			sm_values = np.array(sm_values)
			hp_kernel = hp_kernels.AitchisonAitken(data=self.x_train, bandwidth=bw, num_values=len(self.configspace.get_hyperparameters()[0].choices))
			hp_values = hp_kernel(self.x_test)
			self.assertTrue(np.allclose(hp_values.T, sm_values.squeeze(), 1e-4))



	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_sample(self):
		
		num_samples = 2**20
	
		for bw in [1e-1, 5e-1, 1]:
			hp_kernel = hp_kernels.AitchisonAitken(data=self.x_train, bandwidth=bw, num_values=len(self.configspace.get_hyperparameters()[0].choices))

			samples = hp_kernel.sample(num_samples=num_samples)
			
			
			phat1, phat2 = [], []
		
			for value in [0,1,2]:
				phat1.append(np.sum(samples==value)/num_samples)
				phat2.append(hp_kernel(np.array([value])).mean(axis=0)[0])
			
			for p1, p2 in zip(phat1, phat2):
				self.assertAlmostEqual(p1, p2, delta=5e-3)
				
			self.assertAlmostEqual(np.sum(phat2), 1 , delta=1e-5)




class Test1dInteger(unittest.TestCase):
	n_train = 128
	n_test = 1024
	def setUp(self):
		self.configspace = CS.ConfigurationSpace(43)
		
		HPs=[]
		HPs.append( CS.UniformIntegerHyperparameter('int1', lower=-2, upper=2))
		self.configspace.add_hyperparameters(HPs)
		
		x_train_confs = [ self.configspace.sample_configuration() for i in range(self.n_train)]
		self.x_train = np.array([c.get_array() for c in x_train_confs]).squeeze()	


		x_test_confs = [ self.configspace.sample_configuration() for i in range(self.n_test)]
		self.x_test= np.array(	[c.get_array() for c in x_test_confs]).squeeze()

	def tearDown(self):
		self.configspace = None
		self.x_train = None
		self.x_test = None

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_values(self):
		n = self.configspace.get_hyperparameters()[0].upper - self.configspace.get_hyperparameters()[0].lower + 1

		for bw in [1e-3, 1e-2, 1e-1, 0.99]:
			
			sm_x_train= np.rint(self.x_train* n - .5).astype(np.int)
			sm_x_test = np.rint(self.x_test * n - .5).astype(np.int).squeeze()
			sm_values = np.array([sm_kernels.wang_ryzin(bw, sm_x_train[:,None], x) for x in sm_x_test]).squeeze()
			hp_kernel = hp_kernels.WangRyzinInteger(data=self.x_train, bandwidth=bw, num_values=n, fix_boundary=False)
			hp_values = hp_kernel(self.x_test).squeeze()

			self.assertTrue(np.allclose(hp_values.T, sm_values, 1e-4))


	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_pdf_boundary_quadrature(self):
		self.x_test  = np.array([0,1,2,3,4])/5+(1/10)
		
		for bw in [1e-2, 1e-1, 0.99]:
			hp_kernel = hp_kernels.WangRyzinInteger(data=self.x_train, bandwidth=bw, num_values=5, fix_boundary=True)
			hp_values = hp_kernel(self.x_test).mean(axis=0)
			self.assertAlmostEqual(hp_values.sum(), 1, delta=1e-4)


	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_sample(self):
		num_samples = 2**20
	
		for bw in [1e-1, 5e-1, 0.99]:
			hp_kernel = hp_kernels.WangRyzinInteger(data=self.x_train, bandwidth=bw, num_values=5, fix_boundary=True)

			samples = hp_kernel.sample(num_samples=num_samples)
			phat1, x = np.histogram(samples, normed=True, bins=[0, 0.2, .4, .6, .8, 1.])
			
			phat1 /= 5 # account for bin width
			phat2 = hp_kernel((x[1:] + x[:-1])/2).mean(axis=0)
			
			for p1, p2 in zip(phat1, phat2):
				self.assertAlmostEqual(p1, p2, delta=5e-2)




class Test1dOrdinal(unittest.TestCase):
	n_train = 128
	n_test = 5
	def setUp(self):
		self.configspace = CS.ConfigurationSpace(43)
		
		HPs=[]
		HPs.append( CS.OrdinalHyperparameter('ord1', ['cold', 'mild', 'warm', 'hot']))
		self.configspace.add_hyperparameters(HPs)
		
		x_train_confs = [ self.configspace.sample_configuration() for i in range(self.n_train)]
		
		self.x_train = np.array([c.get_array() for c in x_train_confs]).squeeze()

		x_test_confs = [ self.configspace.sample_configuration() for i in range(self.n_test)]
		self.x_test= np.array(	[c.get_array() for c in x_test_confs]).squeeze()
		
	def tearDown(self):
		self.configspace = None
		self.x_train = None
		self.x_test = None

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_values(self):

		for bw in [1e-3, 1e-2, 1e-1, 1]:
			sm_values = np.array([sm_kernels.wang_ryzin(bw, self.x_train[:,None], x) for x in self.x_test])
			hp_kernel = hp_kernels.WangRyzinOrdinal(data=self.x_train, bandwidth=bw, fix_boundary=False)
			hp_values = hp_kernel(self.x_test)
			self.assertTrue(np.allclose(hp_values.T, sm_values, 1e-4))

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_pdf_boundary_simple(self):
		self.x_train = np.array([0])
		self.x_test  = np.array([0, 1,2,3])
		for bw in [1e-3, 1e-2]:
			# note: for larger bandwidths, the pdf also needs to be truncated as +1,
			# which leads to something different than the scaling computed here
		
			hp_kernel1 = hp_kernels.WangRyzinOrdinal(data=self.x_train, bandwidth=bw, num_values=4, fix_boundary=False)
			hp_kernel2 = hp_kernels.WangRyzinOrdinal(data=self.x_train, bandwidth=bw, num_values=4, fix_boundary=True)
				
			hp_values1 = hp_kernel1(self.x_test).squeeze()
			hp_values2 = hp_kernel2(self.x_test).squeeze()

			weight = 1-hp_values1[1:].sum()
			self.assertTrue(np.allclose(hp_values1/weight, hp_values2, 1e-4))
		
		self.x_train = np.array([3])
		self.x_test  = np.array([0,1,2,3])
		for bw in [1e-3, 1e-2]:
			# note: for larger bandwidths, the pdf also needs to be truncated as +1,
			# which leads to something different than the scaling computed here				

			hp_kernel1 = hp_kernels.WangRyzinOrdinal(data=self.x_train, bandwidth=bw, num_values=4, fix_boundary=False)
			hp_kernel2 = hp_kernels.WangRyzinOrdinal(data=self.x_train, bandwidth=bw, num_values=4, fix_boundary=True)

			hp_values1 = hp_kernel1(self.x_test).squeeze()
			hp_values2 = hp_kernel2(self.x_test).squeeze()

			weight = 1-hp_values1[:-1].sum()
			self.assertTrue(np.allclose(hp_values1/weight, hp_values2, 1e-4))

		
		# simple test based on 68, 95, 99% rule
		self.x_train = np.array([0.5])
		for bw, w in ([0.5, 0.6827], [0.25, 0.9545], [1/6, 0.9973]):
			hp_kernel = hp_kernels.Gaussian(data=self.x_train, bandwidth=bw, fix_boundary=True)
			self.assertAlmostEqual(hp_kernel.weights[0], 1/w, delta=1e-4)

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_pdf_boundary_quadrature(self):
		self.x_test  = np.array([0,1,2,3])
		
		for bw in [1e-2, 1e-1, 0.99]:
			hp_kernel = hp_kernels.WangRyzinOrdinal(data=self.x_train, bandwidth=bw, num_values=4, fix_boundary=True)
			hp_values = hp_kernel(self.x_test).mean(axis=0)
			self.assertAlmostEqual(hp_values.sum(), 1, delta=1e-4)
		

	@unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
	def test_sample(self):
		num_samples = 2**20
	
		for bw in [1e-1, 5e-1, 0.99]:
			hp_kernel = hp_kernels.WangRyzinOrdinal(data=self.x_train, bandwidth=bw, num_values=4, fix_boundary=True)

			samples = hp_kernel.sample(num_samples=num_samples)
			phat1, x = np.histogram(samples, normed=True, bins=[-0.5, 0.5, 1.5, 2.5, 3.5])
			phat2 = hp_kernel((x[1:] + x[:-1])/2).mean(axis=0)
			for p1, p2 in zip(phat1, phat2):
				self.assertAlmostEqual(p1, p2, delta=5e-2)
		
		
if __name__ == '__main__':
	unittest.main()
