

Quickstart Guide
================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:

What is HpBandSter?
~~~~~~~~~~~~~~~~~~~

HpBandSter (HyperBand on STERoids) implements recently published methods for
optimizing hyperparameters of machine learning algorithms. We designed
HpBandSter such that it scales from running sequentially on a local machine
to running on a distributed system in parallel

One of the implemented algorithms is **BOHB**, which combines Bayesian
Optimization and HyperBand to efficiently search for well performing configurations.
Learn more about this method by reading out paper, published at
`ICML 2018 <http://proceedings.mlr.press/v80/falkner18a.html>`_


How to install HpBandSter
~~~~~~~~~~~~~~~~~~~~~~~~~

HpBandSter can be installed via pip for python3:

.. code-block:: bash

    pip install hpbandster

If you want to develop on the code you could install it via

.. code-block:: bash

    python3 setup.py develop --user

.. note::

    HpBandSter is only supported for python3

How to use HpBandSter
~~~~~~~~~~~~~~~~~~~~~

To get started, we will guide you through some basic examples:

1) :ref:`Basic Setup: Local and Sequential Usage <1st example>`
2) :ref:`Advanced: Distributed and Parallel Usage <2nd example>`

In the :doc:`advanced examples <advanced_examples>`, there will be shown

3) :ref:`Continue Runs and Visualize Results <3rd example>`
4) :ref:`Combine BOHB and CAVE to analyze results <BOHB with CAVE>`

.. note::

    For some further examples, please visit the :doc:`gallery <auto_examples/index>`.

.. _1st example:

Local and Sequential
~~~~~~~~~~~~~~~~~~~~

Whether you like to use BOHB locally on your machine or on a cluster, the setup
always consists of three ingredients: The *master*, also refered to a the
optimization algorithm, steers the hyperparameter optimization
and communicates over the *nameserver* with *workers* to use them to evaluate configurations:

:ref:`Configure and Set up a Nameserver`
   | The *nameserver* serves as a phonebook-like lookup table keeping track and communicating
     with the *workers*. Unique names are created so the *workers* can work in parallel and register their results
     without creating racing conditions. It manages also the communication between all *workers*.

:ref:`Implement and Instantiate a Worker`
   | The *worker* is responsible for evaluating a given model with a single configuration on a single budget at a time.

:ref:`Initialize and Start the Master`
   | The *master* (here: :py:class:`BOHB <hpbandster.optimizers.bohb>`) is
     responsible for book keeping and decides which configuration the workers should evaluate next.
     Optimizers are instantiations of the *master*-class, that handle the important steps of deciding what
     configurations to run on what budget.

.. _Configure and Set up a Nameserver:

Step 1: Set up a :py:class:`Nameserver <hpbandster.core.nameserver>`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

| First, we start the :py:class:`nameserver <hpbandster.core.nameserver>` and assign a *run_id* to it.
  The *run_id* can be a string or an integer to for example describe the current experiment. Since
  we work locally, we can pass a random port (here: 0) to the NameServer.

.. literalinclude:: ../../hpbandster/examples/example_1_simple_locally.py
    :lines: 29-30, 40-44

.. note:: The *run_id* has to be unique for concurrent runs, i.e. when multiple optimization runs are executed
   at the same time, they have to have different *run_id*'s.

.. _Implement and Instantiate a Worker:

Step 2: Implement and Instantiate a :py:class:`Worker <hpbandster.core.worker>`
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

| Next, we set up a *worker*. The *worker* is responsible to evaluate a hyperparameter setting.
  The *worker* requires the *nameserver*, its *port*, as well as the *run_id* to carry out evaluations.
  In this first example, we work with only one *worker*. How to use more than one
  locally or on a cluster will be explained in a later example
  (e.g. :doc:`Example 5 <auto_examples/example_5_mnist>`)

.. literalinclude:: ../../hpbandster/examples/example_1_simple_locally.py
    :lines: 52-56

The *worker* **inherits** from the class :py:class:`hpbandster.core.worker`
and **overwrites the compute-method** .
The *compute* -method will be called by the optimizer (master) with a
configurations and must return the computed loss (and optionally additional information).

Here's a short excerpt from the *worker* used in (:doc:`example 1 <auto_examples/commons>`)

.. literalinclude:: ../../hpbandster/examples/commons.py
    :lines: 15-16, 18-53


| Before we can continue to *Step 3*, we have to create a *ConfigurationSpace*-object defining
  the space of possible hyperparameters and their ranges to search during optimization.
  For this we make use of the ConfigSpace_-package and define one continuous hyperparameter:

.. literalinclude:: ../../hpbandster/examples/commons.py
    :lines: 56, 62-64

.. note::
    Of course, we also support categorical and conditional hyperparameter types.
    For more examples we refer to the documentation of the ConfigSpace_ or
    please have a look at the :doc:`ConfigSpace example<auto_examples/example_4_config_space>`.

.. note::
    It's good practice to save the configuration space to file, so that you can use it later in
    analysis tools like CAVE_ .

.. _Initialize and Start the Master:

Step 3: Initialize and Start the Master
++++++++++++++++++++++++++++++++++++++++

| Finally, we can can instantiate a *master*. In this example we will use *BOHB*,
  for alternatives, see :doc:`here <optimizers>`.
  It performs iterative rounds of Successive Halving while in each round proposing
  a set of configurations using Bayesian optimizations.
| Besides the hyperparameters *eta*, *min_budget* and *max_budget*, the optimizer also requires
  a *run_id* (must be the same as for the *nameserver*), a reference to the *nameserver*, a *port* and of course
  the *configspace*.

.. literalinclude:: ../../hpbandster/examples/example_1_simple_locally.py
    :lines: 68-75

| First, BOHB will evaluate a set of configuration with the *min_budget* and then eliminates the 1 / *eta* worst
  performing configurations. Then, the same time the budget for the next round will be increased by a factor of *eta*.
| This process runs until the maximum budget is reached and one configuration is left.
  So *eta* not only defines the elimination ratio, but also the size of the initial set of configurations.

.. note:: For example, if *eta* = 2, *min_budget* = 1, *max_budget* =10,
  then in the first round we have 8 configurations with a budget of 1.
  Only 8/2=4 configurations will advance to the second iteration with a budget of 1*2=2.
  In the 3. iteration, there will be 2 configurations left and in the
  last iteration only one configuration will run with the *max_budget* of 10.

When everything is set up, the optimizer can be started.
*n_iterations* specify how many iterations BOHB will run.
After it has finished, the *master* will shut down.

.. literalinclude:: ../../hpbandster/examples/example_1_simple_locally.py
    :lines: 79, 82

The optimization-run returns an :py:class:`result object <hpbandster.core.result>`.
This class offers a simple API to access those information.

For example, we can plot the incumbent trajectory

.. literalinclude:: ../../hpbandster/examples/example_1_simple_locally.py
    :lines: 88-90, 96-102

or access the best found configuration:

.. code-block:: python

    incumbent_id = res.get_incumbent_id()
    incumbent_config = id2config[incumbent_id]['config']

.. _2nd example:

Distributed and Parallel
~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will show how to use HpBandSter on a cluster

| :ref:`Initialize Nameserver, Master, First Worker`
| :ref:`Initialize more Workers`
| :ref:`Submit the Job` to the cluster

The workflow to use HpBandster on a cluster is similar to example 1. We first
start a *nameserver*, a *master*, and multiple *workers*.
This time we start the example via bash as shown in Step 3.

.. _Initialize Nameserver, Master, First Worker:

1) Initialize Nameserver, Master, First Worker
++++++++++++++++++++++++++++++++++++++++++++++

On the first node (*array_id == 1*), we start the *nameserver*, and the *master*.
BOHB is usually so cheap, that we can afford to run a *worker* on the *master* node, too.

.. literalinclude:: ../../hpbandster/examples/example_2_cluster.py
    :lines: 23-30, 37, 39-46, 48-66, 68, 71


.. _Initialize more Workers:

2) Initialize more Workers
++++++++++++++++++++++++++

The other *workers*, which will be run on other nodes only instantiate the *worker*-class, connect to the
*nameserver* and start serving.

.. literalinclude:: ../../hpbandster/examples/example_2_cluster.py
    :lines: 73, 74-76, 78-80, 82


.. _Submit the Job:

3) Submit the Job
+++++++++++++++++

We start our optimization run with a *bash file* to submit our jobs to a
resource manager (*here SunGridEngine*).

.. literalinclude:: ../../hpbandster/examples/example_2_cluster_submit_me.sh


.. _ConfigSpace: https://github.com/automl/ConfigSpace
.. _CAVE: https://github.com/automl/CAVE
