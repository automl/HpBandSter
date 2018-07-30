

Quickstart Guide
================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:

What is HpBandSter?
~~~~~~~~~~~~~~~~~~

HpBandSter (HyperBand on STERoids) implements recently published methods for
optimizing hyperparameters of machine learning algorithms. We designed
HpBandSter such that it scales from running sequentially on a local machine
to running on a distributed system in parallel

One of the implemented algorithms is **BOHB**, which combines Bayesian
Optimization and HyperBand to efficiently search for well performing configurations.
Learn more about this method by reading out paper, published at `ICML 2018 <https://arxiv.org/pdf/1807.01774.pdf>`_

How to use HpBandSter
~~~~~~~~~~~~~~~~~~~~~

To get started, we will guide you through some basic examples:

1) :ref:`Basic Setup: Local and Sequential Usage <1st example>`
2) :ref:`Advanced: Distributed and Parallel Usage <2nd example>`
3) :ref:`Continue Runs and Visualize Results <3rd example>`
4) :ref:`Combine BOHB and CAVE to analyze results <BOHB with CAVE>`

.. note::

    For some further examples, please visit the :doc:`gallery <auto_examples/index>`.

.. _1st example:

1st example - Local and Sequential Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether you like to use BOHB locally on your machine or on a cluster, the setup
always consists of three ingredients: The *master*, also refered to a the
optimization algoritm, steers the hyperparameter optimization
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
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
  for althernatives, see :doc:`here <optimizers>`.
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

2nd example - Distributed and Parallel Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will show how to use HpBandSter on a cluster

| :ref:`Initialize Nameserver, Master, First Worker`
| :ref:`Initialize more Workers`
| :ref:`Submit the Job` to the cluster

The workflow to use HpBandster on a cluster is similar to example 1. We first
start a *nameserver*, a *master*, and multiple *workers*.
This time we start the example via bash as shown in Step 3.

.. _Initialize Nameserver, Master, First Worker:

1) Initialize Nameserver, Master, First Worker
+++++++++++++++++++++++++++++++++++

On the first node (*array_id == 1*), we start the *nameserver*, and the *master*.
BOHB is usually so cheap, that we can afford to run a *worker* on the *master* node, too.

.. literalinclude:: ../../hpbandster/examples/example_2_cluster.py
    :lines: 23-30, 37, 39-46, 48-66, 68, 71


.. _Initialize more Workers:

2) Initialize more Workers
+++++++++++++++++++++++++++++++

The other *workers*, which will be run on other nodes only instantiate the *worker*-class, connect to the
*nameserver* and start serving.

.. literalinclude:: ../../hpbandster/examples/example_2_cluster.py
    :lines: 73, 74-76, 78-80, 82


.. _Submitting the job:

3) Submit the Job
+++++++++++++++++++++

We start our optimization run with a *bash file* to submit our jobs to a
resource manager (*here SunGridEngine*).

.. literalinclude:: ../../hpbandster/examples/example_2_cluster_submit_me.sh

.. _3rd example:

3rd example - Continue Runs and Visualize Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example we will cover the functionalities:

| :ref:`live logging`
| :ref:`warmstart`
| :ref:`warmstart visualization`

.. _live logging:

1) Live logging
+++++++++++++++

We'd like to introduce another function: **live logging**.

HpBandSter is able to write all results on the fly to file.
This is done by using a :py:class:`result logger <hpbandster.core.result.json_result_logger>` .
It receives as input a storage path and the boolean parameter *overwrite*.
If *overwrite* is set to true, already existing results will be replaced by the new ones.
Otherwise the result logger will just append the new results to the existing file.

Additionally, we set up again the *nameserver*, *worker* and the *master*.

.. literalinclude:: ../../hpbandster/examples/example_3_warmstarting_visualization.py
    :lines: 22-29, 34-39, 43, 50-51, 58-59, 61-64, 73-82

The parameter *min_points_in_model* specifies the number of observations BOHB needs to start building a *kernel density estimation* (KDE).
The default value is *dim+1*.

The parameter *min_bandwidth* forces the model to keep diversity. Even when all samples have the same value
for one of the parameters, a minimum bandwidth (Default: 1e-3) is used instead of zero.

.. _warmstart:

2) Warmstart
++++++++++++
We start the *master* for the first time and after the run we shutdown the *master*, but keep the *nameserver* alive:

.. literalinclude:: ../../hpbandster/examples/example_3_warmstarting_visualization.py
    :lines: 85-86

| We increase the minimum and maximum budget by a factor of 3.
| Now let's start a new run, but feed in the results of the first one to warmstart the model.
  Note that the budgets don't have to align, but beware: if the maximum budget of the second run is not
  greater or equal to the maximum budget in the previous runs, BOHB's model might never be updated!

.. literalinclude:: ../../hpbandster/examples/example_3_warmstarting_visualization.py
    :lines: 92-108, 110-111

.. _warmstart visualization:

3) Warmstart visualization
++++++++++++++++++++++++++

HpBandster contains an interactive visualization tool to plot the results of the optimization run.
We use it here to show the warmstart functionality.

.. literalinclude::  ../../hpbandster/examples/example_3_warmstarting_visualization.py
    :lines: 124-133

.. _BOHB with CAVE:

4th example - Combine BOHB and CAVE to analyze results
~~~~~~~~~~~~~~

To run CAVE on BOHB-results, you need a folder with the files *results.json*, *configs.json* and *configspace.pcs* in
it. CAVE will output an individual report for each budget. Simply run:

    .. code-block:: bash

        cave --folder bohb_results_folder --ta_exec_dir folder_from_which_bohb_was_run --file_format BOHB

E.g. to analyze the RNN-example, just run:

    .. code-block:: bash

        python hpbandster/examples/example_6_rnn_20_newsgroups.py
        cave --folder results_example_rnn --ta_exec_dir . --file_format BOHB

.. note::

        To use CAVE with BOHB, currently you have to install CAVE from the development-branch
        (e.g. `pip install git+https://github.com/automl/CAVE@development`).


.. _ConfigSpace: https://github.com/automl/ConfigSpace
.. _CAVE: https://github.com/automl/CAVE
