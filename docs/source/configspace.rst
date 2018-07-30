ConfigSpace
===========

In this example, we'll show the example use of the ConfigSpace-Module

It is for example used in our optimizer (e.g. SMAC or BOHB).
A ConfigSpace object organizes the hyperparamters to be optimized.
It offers the functionality to sample configurations from the defined configuration space.

It is also simple to distinguish between different hyperparameter types, like integer, float or categorical
hyperparamters.

A powerful advantage in comparison to naive implementations is the ability to realize conditional hyperparameters.


In this tutorial, you'll see how to:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`Create a configuration space` and add
    | :ref:`Float hyperparameter`
    | :ref:`Categorical hyperparamter`
    | :ref:`Integer hyperparameter`

:ref:`2) Realize conditions with conditional hyperparameter <Conditional hyperparameters>`

:ref:`neural network`

:ref:`worker`

:ref:`bohb`

For demonstration purpose we'll train a one to three hidden layer neural network with either adam or sgd optimizer.

**So we'll optimise the following hyperparameters:**

- learning rate:             (float)        [1e-6, 1e-2]
- activation funct           (categorical)  ['ReLU', 'Tanh']
- optimizer:                 (categorical)  ['Adam', 'SGD']

	+ sgd momentum:            (float)        [0.0, 0.99]        # only if optimizer == 'SGD'
- number of hidden layers    (int)          [1,3]

	+ dimension hidden layer 1 (int)          [100, 1000]
	+ dimension hidden layer 2 (int)          [100, 1000]        # only if number of hidden layers > 1
	+ dimension hidden layer 3 (int)          [100, 1000]        # only if number of hidden layers > 2

.. _Create a configuration space:

1) Create a configurations space
--------------------------------

At the beginning, we create a *ConfigurationSpace*-object.
It is equal to a container holding all different kinds of hyperparameters.

.. literalinclude:: ../../hpbandster/examples/example_4_config_space_worker.py
    :dedent: 4
    :lines: 128-129, 135

As already mentioned, *ConfigSpace* supports different types of Hyperparamters.

.. _Float hyperparameter:

1.1) Float hyperparameter
+++++++++++++++++++++++++

| We add the learning rate-hyperparameter to the config space.
  Its values should fall within a range of 1e-6, 1e-2. If the *log* parameter is set to true,
  ConfigSpace will sample the values on a logarithmic scale.

.. literalinclude:: ../../hpbandster/examples/example_4_config_space_worker.py
    :dedent: 4
    :lines: 137-138

.. _Categorical hyperparamter:

1.2) Categorical hyperparamter
++++++++++++++++++++++++++++++

|  We add also some categorical hyperparameters to the *ConfigSpace*.
  (here: activation function)

.. literalinclude:: ../../hpbandster/examples/example_4_config_space_worker.py
    :dedent: 4
    :lines: 139-140

.. _Integer hyperparameter:

1.3) Integer hyperparameter
+++++++++++++++++++++++++++

| In this example here, number of hidden layers is a integer hyperparamter,
  the dimensions of each layer.

.. literalinclude:: ../../hpbandster/examples/example_4_config_space_worker.py
    :dedent: 4
    :lines: 162-173

.. _Conditional hyperparameters:

2) Conditional hyperparameter
-----------------------------
| After you have seen, how to use the basic types, we give an example of conditional hyperparameters.
| We choose between two different optimizers. *Stochastic Gradient Descent* (=SGD) and Adam.
| For demonstration purpose, we give *SGD* another hyperparameter, *sgd_momentum*,
  which should only be activated if the hyperparameter *optimizer* is equal to *SGD*.
| This can be realized with conditions. In this case with a *EqualsCondition*.

.. literalinclude:: ../../hpbandster/examples/example_4_config_space_worker.py
    :dedent: 4
    :lines: 145-153, 156-157

| It is also possible to realize inequality constraints.
| In the code, you can find inequality constraints on the dimensions of the hidden layers.
| We want the *hidden_dim_2* hyperparameter only to be activated if *num_hidden_layers* is greater than 1.
  Analogous to this, *hidden_dim_3* should only be activated if *num_hidden_layers* is greater than 2.

.. literalinclude:: ../../hpbandster/examples/example_4_config_space_worker.py
    :dedent: 4
    :lines: 176-182

.. _neural network:

3) Create a neural network
--------------------------

As neural network we use a simple pytorch implementation of a feed forward network.

.. literalinclude:: ../../hpbandster/examples/example_4_config_space_worker.py
    :lines: 185, 189-219


.. _worker:

4) Create a worker
------------------

In the *__init__* we load the data and split it into training and test set.

.. literalinclude:: ../../hpbandster/examples/example_4_config_space_worker.py
    :lines: 45-67

The *compute* method runs the network with the given hyperparameters. This function is called by
the *master* (here: :py:class:`BOHB <hpbandster.optimizers.bohb>`).
The sampled configurations are stored in the *config* - dictionary.

.. literalinclude:: ../../hpbandster/examples/example_4_config_space_worker.py
    :dedent: 4
    :lines: 69, 88-126


.. _bohb:

5) Run BOHB
-----------

For reasons of completion, we'll use again BOHB to optimize the hyperparameters.
How this workers is already explained :doc:`here <quickstart>`

.. literalinclude:: ../../hpbandster/examples/example_4_config_space.py
    :lines: 14-15, 24-25, 31-33, 39-44, 57-65, 69-71
