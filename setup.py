from setuptools import setup

setup(
	name='hpbandster',
	version='0.1.0',
	description='HyPerBAND on STERoids, a parallel Hyperband implementation with lots of room for improvement',
	author='Stefan Falkner',
	author_email='sfalkner@cs.uni-freiburg.de',
	license='Use as you wish. No guarantees whatsoever.',
	classifiers=['Development Status :: 3 - Alpha'],
	packages=['hpbandster'],
	python_requires='>=3',
	install_requires=['Pyro4', 'serpent', 'ConfigSpace']
)
