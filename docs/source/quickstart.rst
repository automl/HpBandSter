

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

	git clone git@github.com:automl/HpBandSter.git
	cd HpBandSter
    python3 setup.py develop --user

.. note::

    We only support Python3 for HpBandSter!

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

The basic Ingredients
~~~~~~~~~~~~~~~~~~~~~

Whether you like to use HpBandSter locally on your machine or on a cluster, the basic setup
is always the same. For now, let's focus on the most important ingredients needed
to apply an optimizer to a new problem:

:ref:`Implement a Worker`
   | The *worker* is responsible for evaluating a given model with a single configuration on a single budget at a time.

:ref:`Define the Search Space`
   | Next, the parameters being optimized need to be defined. HpBandSter relies on the **ConfigSpace** package for that.

:ref:`Pick the Budgets and the Number of Iterations`
   | To get good performance, HpBandSter needs to know meaningful budgets to use. You also have to specify how many iterations the optimizer performs.

1. A :py:class:`Worker <hpbandster.core.worker>`
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

| First, we need to implement a *worker* for the problem.
  The *worker* is responsible to evaluate a hyperparameter setting and returning the associated loss that is minimized.
  By deriving from the :py:class:`base class <hpbandster.core.worker>`, encoding a new problem consists of implementing two methods: **__init__** and **compute**.
  The first allows to perform inital computations, e.g. loading the dataset, when the worker is started, while the latter is called repeatedly called during the optimization and evaluates a given configuration yielding the associated loss.

| The worker below demonstrates the concept.
  The worker implements a simple toy problem where there is a single parameter `x` in the configuration and we try to minimize it.
  The function evaluations are corrupted by some Gaussian noise that shrinks as the budget grows.

.. literalinclude:: ../../hpbandster/examples/commons.py
    :lines: 8-50


2. The Search Space Definition
++++++++++++++++++++++++++++++

| Every problem needs a description of the search space to be complete.
  In HpBandSter, a *ConfigurationSpace*-object defining all hyperparameters, their ranges, and potential dependencies between them.
  In our toy example here, the search space consists of a single continuous parameter `x` between zero and one.
  For convenience, we attach the configuration space definition to the worker as a static method.
  This way, the worker's compute function and its parameters are neatly combined.

.. literalinclude:: ../../hpbandster/examples/commons.py
    :lines: 14-15, 48-53

.. note::
    Of course, we also support integer, ordinal, and  categorical hyperparameters.
    To express dependencies, the ConfigSpace package also also to express conditions and forbidden relations between parameters.
    For more examples we refer to the documentation of the ConfigSpace or
    please have a look at the :doc:`ConfigSpace example<auto_examples/example_4_config_space>`.


3. Meaningful Budgets and Number of Iterations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

| To take advantage of lower fidelity approximation, i.e. budgets lower than *max_budget*, those *lower accuracy* evaluations have to be meaningful.
  As these budgets can mean very different things (epochs of training a neural network, number of data points to train the model, or number of cross-validation folds to name a few), these have to be user specified.
  This is done by two parameters, called `min_budget` and `max_budget` for all optimizers.
  For better speed ups, the lower budget should be as small as possible while still being informative.
  By `informative`, we mean that the performance is a *ok* indicator for the loss on higher budgets.
  It's hard to be more concrete for the general case.
  The two budgets are problem dependent and require some domain knowledge.


| The number of iterations is usually a much easier parameter to pick.
  Depending on the optimizer, an iteration requires the computational budget of a couple of function evaluations on `max_budget`
  In general the more the better, and things become more complicated when multiple workers run in parallel.
  For now, the number of iterations simply controls how many configurations are evaluated.




The first toy examples
~~~~~~~~~~~~~~~~~~~~~~

| Let us now take the above worker, its search space and that in a few different settings.
  Specifically, we will run

1. locally and sequentially
2. locally and in parallel (thread based)
3. locally and in parallel (process based)
4. distributed in a cluster environment

| Each example, showcases how to setup HpBandSter in different environments and highlights specifics for it.
  Every compute environment is slightly different, but it should be easy to bootstrap from one of the examples and adapt it to any specific needs.
  The first example slowly introduces the main workflow for any HpBandSter run.
  the following ones gradually add complexity by including more features.



1. A Local and Sequential Run
+++++++++++++++++++++++++++++
| We are now ready to look at our first real example [ADD LINK!!] to illustrate how HpBandSter is used.
  Every run consists of the same 5 basic steps which we will now cover.

Step 1: Start a :py:class:`Nameserver <hpbandster.core.nameserver.NameServer>` 

.. literalinclude:: ../../hpbandster/examples/example_1_local_sequential.py
    :lines: 31-32

Step 2: Start a :py:class:`Worker <hpbandster.core.worker.Worker>` 

.. literalinclude:: ../../hpbandster/examples/example_1_local_sequential.py
    :lines: 39-40

Step 3: Run an Optimizer :py:class:`Worker <hpbandster.core.master.Master>` 

.. literalinclude:: ../../hpbandster/examples/example_1_local_sequential.py
    :lines: 46-50

Step 4: Stop all services

.. literalinclude:: ../../hpbandster/examples/example_1_local_sequential.py
    :lines: 54-55


Step 5: Analysis of the Results

| After a run is finished, one might be interested in all kinds of information.
  HpBandSter offers full access to all evaluated configurations including timing information and potential error messages for failed runs.
  In this first example, we simply look up the best configuration (called incumbent), count the number of configurations and evaluations, and the total budget spent.
  For more details, see some of the other examples and the documentation of the :py:class:`Result <hpbandster.core.result.Result>` class.

.. literalinclude:: ../../hpbandster/examples/example_1_local_sequential.py
    :lines: 62-69

| The complete source code for this example can be found here [ADD LINK!!].


2. A Local parallel Run using Threads
+++++++++++++++++++++++++++++++++++++

| Let us now extend this example to start multiple workers, each in a separate thread.
  This is a useful mode to exploit a multicore CPU system, if the individual workers get around Python's global interpreter lock.
  For example, many scikit learn algorithms outsource the heavy duty computations to some C module, making them run truly in parallel even if threaded.

| Below, we can instantiate the specified number of workers.
  To emphasize the effect, we introduce a sleep_interval of one second, which makes every function evaluation take a bit of time.
  Note the additional id argument that helps separating the individual workers.
  This is necessary because every worker uses its processes ID which is the same for all threads here.

.. literalinclude:: ../../hpbandster/examples/example_2_local_parallel.py
    :lines: 38-42


| When starting the optimizer, we can add the min_n_workers argument to the run methods to make the optimizer wait for all workers to start.
  This is not mandatory, and workers can be added at any time, but if the timing of the run is essential, this can be used to synchronize all workers right at the start.

.. literalinclude:: ../../hpbandster/examples/example_2_local_parallel.py
    :lines: 52
