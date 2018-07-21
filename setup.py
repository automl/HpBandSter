from setuptools import setup, find_packages

setup(
	name='hpbandster',
	version='0.6.0',
	description='HyPerBAND on STERoids, a distributed Hyperband implementation with lots of room for improvement',
	author='Stefan Falkner',
	author_email='sfalkner@cs.uni-freiburg.de',
	url="https://github.com/automl/HpBandSter",
	license='BSD 3-Clause License',
	classifiers=['Development Status :: 4 - Beta'],
	packages=find_packages(),
	python_requires='>=3',
	install_requires=['Pyro4', 'serpent', 'ConfigSpace', 'numpy','statsmodels', 'scipy'],
	extras_require = {
		'Automatic hostname inference': ['netifaces'],
	},
	keywords=['distributed', 'optimization', 'multifidelity'],
)
