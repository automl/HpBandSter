

Quickstart Guide
================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:

What is HpBandSter
~~~~~~~~~~~~~~~~~~

a distributed Hyperband implementation on Steroids.

It contains **BOHB**, which combines Bayesian Optimization and HyperBand.
The paper can be found `here <https://arxiv.org/pdf/1807.01774.pdf>`_

HpBandSter is able to optimize hyperparameters locally or on a cluster, sequential or parallel.

How to use HpBandSter
~~~~~~~~~~~~~~~~~~~~~

To get started, we will guide you through some basic examples.
The full examples are located in the :doc:`gallery <auto_examples/index>`.

1) :ref:`we will explain the basic usage of BOHB by optimizing a toy function <1st example>`
2) :ref:`we will expand example 1 to use it on a cluster <2nd example>`
3) :ref:`we will show you how to do warmstarting with BOHB and how to use the visualization tool <3rd example>`
4) :ref:`we will give you an introduction how to combine BOHB with CAVE_ to analyze the results <BOHB with CAVE>`

.. note::

    For some further examples, please visit the :doc:`gallery <auto_examples/index>`.

.. _1st example:

1st example - local, sequential evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the first example, you'll see the basic usage of BOHB. 
It'll run locally with a simple setup.

Whether you like to use BOHB on your machine or on a cluster,
basically you need just to follow the next three steps:

:ref:`Set up a nameserver`
   | The *nameserver* serves as a phonebook-like lookup table for your *workers*.
     Unique names are created so the *workers* can work in parallel and register their results
     without creating racing conditions.
   | It manages also the communication between all *workers*.

:ref:`Implement a worker`
   | The *worker* is responsible for evaluating a given model with a single configuration on a single budget at a time.

:ref:`Initialize the master & start it`
   | The *master* (here: :py:class:`BOHB <hpbandster.optimizers.bohb>`) is responsible for book keeping and to decide what to run next.
   | For example, it samples the configuration to evaluate and passes it to a free *worker*.
   | Optimizers are instantiations of the *master*-class, that handle the important steps of deciding what
     configurations to run on what budget.

.. _Set up a nameserver:

Step 1: Set up a :py:class:`nameserver <hpbandster.core.nameserver>`
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

| First, we start the :py:class:`nameserver <hpbandster.core.nameserver>` and assign a *run_id* to it.
  This *run_id* has to be unique for concurrent runs, i.e. when multiple instances run at the same time,
  they have to have different *run_id* s.
| Since we work locally, we can pass a random port (here: 0) to the NameServer.

.. literalinclude:: ../../hpbandster/examples/example_1_simple_locally.py
    :lines: 29-30, 40-44


.. _Implement a worker:

Step 2: Implement a :py:class:`worker <hpbandster.core.worker>`
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

| Next, we have to set up a *worker*. The *worker* implements the connection to the model to be evaluated.
| It needs a reference to the *nameserver* and its port, as well as the *run_id*.
| In this first example, we work with only one *worker*. How to use more than one
  locally or on a cluster will be explained in a later example
  (e.g. :doc:`Example 5 <auto_examples/example_5_mnist>`)

.. literalinclude:: ../../hpbandster/examples/example_1_simple_locally.py
    :lines: 52-56

In the *worker*, we need to **inherit** from the class :py:class:`hpbandster.core.worker`
and **overwrite the compute-method** .
Its *compute* -method will be called later by the BOHB-optimizer repeatedly
with the sampled configurations and return the computed loss (and additional infos).

Here's a short excerpt from the *worker* used in example 1. (:doc:`Worker <auto_examples/commons>`)

.. literalinclude:: ../../hpbandster/examples/commons.py
    :lines: 15-16, 18-53


| Before we can continue to *Step 3*, we have to create a ConfigSpace_-object.
  It contains the hyperparameters to be optimized.

.. literalinclude:: ../../hpbandster/examples/commons.py
    :lines: 56, 62-66

.. note::
    It's good practice to save the configuration space to file, so that you can use it later in
    tools like CAVE_ .

| For more insights into the ConfigSpace, please have a look in the
  :doc:`ConfigSpace-example <auto_examples/example_4_config_space>`
| In the near future, there will be a documentation for the ConfigSpace_ .

.. _Initialize the master & start it:

Step 3: Initialize the master & start it
++++++++++++++++++++++++++++++++++++++++

| In the last of the three steps, we create a *master*, or also referred to as :doc:`optimizer object<optimizers>`.
  It samples configurations from the ConfigurationSpace, using successive halving.
| The number of sampled configurations is determined by the
  parameters *eta*, *min_budget* and *max_budget*.
| After evaluating each configuration, starting with the minimum budget
  on the same subset size, only a fraction of 1 / *eta* of them
  'advances' to the next round. At the same time the current budget will be doubled.
| This process runs until the maximum budget is reached.

| For example, if *eta* = 2, *min_budget* = 1, *max_budget* =10,
  then in the first round we have 8 configurations with a budget of 1.
  Only 4 configurations will advance to the second iteration with a budget of 2.
  In the 3. iteration, there will be 2 configurations with a budget of 5.
  And in the last iteration only one configuration will run with a maximum budget of 10.

We need to pass the configuration space, as well as *nameserver* information to the object.

.. literalinclude:: ../../hpbandster/examples/example_1_simple_locally.py
    :lines: 68-75

When everything is set up, the optimizer can be started.
With the parameter *n_iterations*, it can be specified how many iterations BOHB will run.
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

2nd example - HpBandSter on a cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we will show how to use HpBandSter on a cluster

| :ref:`Nameserver, master, first worker`
| :ref:`Setting up the other workers`
| :ref:`Submitting the job` to the cluster

The workflow to use HpBandster on a cluster is similar to example 1. We have to start a *nameserver*, a *master*, and
in comparison to example 1 multiple *workers*.
This time we call the script with a bash script. It will be shown in Step 3.

.. _Nameserver, master, first worker:

1) Nameserver, master, first worker
+++++++++++++++++++++++++++++++++++

On the first node (*array_id == 1*), we start the *nameserver*, and the *master*.
BOHB is usually so cheap, that we can affort to run a *worker* on the *master* node, too.

.. literalinclude:: ../../hpbandster/examples/example_2_cluster.py
    :lines: 23-30, 37, 39-46, 48-66, 68, 71


.. _Setting up the other workers:

2) Setting up the other workers
+++++++++++++++++++++++++++++++

The other *workers*, which will be started on other nodes. They only instantiate the *worker*-class, look for the
*nameserver* and start serving.

.. literalinclude:: ../../hpbandster/examples/example_2_cluster.py
    :lines: 73, 74-76, 78-80, 82


.. _Submitting the job:

3) Submitting the job
+++++++++++++++++++++

We start our optimization run with a *bash file* to submit our jobs to a
resource manager (*here SunGridEngine*).

.. literalinclude:: ../../hpbandster/examples/example_2_cluster_submit_me.sh

.. _3rd example:

3rd example - Warmstarting
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

BOHB with CAVE
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
