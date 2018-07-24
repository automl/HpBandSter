ConfigSpace
===========

In this example, we'll show the example use of the ConfigSpace-Module

It is for example used in our optimizer (e.g. SMAC or BOHB).
A ConfigSpace object organizes the hyperparamters to be optimized.
It offers the functionality to sample configurations from the defined configurationspace.

It is also simple to distinguish between different hyperparameter types, like integer, float or categorical
hyperparamters.

A powerful advantage in comparison to naive implementations is the ability to realize conditional hyperparameters.


**In this tutorial, you'll see how to:**

- connect a worker with a neural network
- create a configurations space
- add hyperparameters of types float, integer and categorical to the ConfigSpace
- how to create conditional hyperparameters

For demonstration purpose we'll train a one to three hidden layer neural network with either adam or sgd optimizer.


**So we'll optimise the following hyperparameters:**

- learning rate:             (float)        [1e-6, 1e-2]
- optimizer:                 (categorical)  ['Adam', 'SGD']
	- sgd momentum:            (float)        [0.0, 0.99]        # only if optimizer == 'SGD'
- number of hidden layers    (int)          [1,3]
	- dimension hidden layer 1 (int)          [100, 1000]
	- dimension hidden layer 2 (int)          [100, 1000]        # only if number of hidden layers > 1
	- dimension hidden layer 3 (int)          [100, 1000]        # only if number of hidden layers > 2

.. literalinclude:: ../../hpbandster/examples/example_config_space/worker_mnist.py
	:lines: 126-179

.. literalinclude:: ../../hpbandster/examples/example_config_space/worker_mnist.py
	:lines: 44-124, 181-214
	
For reasons of completion, we'll use again BOHB to optimize the hyperparameters.

.. literalinclude:: ../../hpbandster/examples/example_config_space/example_mnist.py
