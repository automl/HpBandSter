

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

The basic Ingredients
~~~~~~~~~~~~~~~~~~~~~

Whether you like to use HpBandSter locally on your machine or on a cluster, the basic setup
is always the same. For now, let's focus on the most important ingredients needed
to apply an optimizer to a new problem:

:ref:`Implementing a Worker<worker>`
   | The *worker* is responsible for evaluating a given model with a single configuration on a single budget at a time.

:ref:`Defining the Search Space<searchspace>`

   | Next, the parameters being optimized need to be defined. HpBandSter relies on the **ConfigSpace** package for that.

:ref:`Picking the Budgets and the Number of Iterations<budgets>`
   | To get good performance, HpBandSter needs to know meaningful budgets to use. You also have to specify how many iterations the optimizer performs.

.. _worker:

1. Implementing a Worker
++++++++++++++++++++++++

| The :py:class:`Worker <hpbandster.core.worker>` is responsible to evaluate a hyperparameter setting and returning the associated loss that is minimized.
  By deriving from the :py:class:`base class <hpbandster.core.worker>`, encoding a new problem consists of implementing two methods: **__init__** and **compute**.
  The first allows to perform inital computations, e.g. loading the dataset, when the worker is started, while the latter is called repeatedly called during the optimization and evaluates a given configuration yielding the associated loss.

| The worker below demonstrates the concept.
  It implements a simple toy problem where there is a single parameter `x` in the configuration and we try to minimize it.
  The function evaluations are corrupted by some Gaussian noise that shrinks as the budget grows.

.. literalinclude:: ../../hpbandster/examples/commons.py
    :lines: 8-50

.. _searchspace:

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
    We also support integer and  categorical hyperparameters.
    To express dependencies, the ConfigSpace package also also to express conditions and forbidden relations between parameters.
    For more examples we refer to the documentation of the ConfigSpace or
    please have a look at the :doc:`advanced_examples`.

.. _budgets:

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

1. :ref:`locally and sequentially<example 1>`
2. :ref:`locally and in parallel (thread based)<example 2>`
3. :ref:`locally and in parallel (process based)<example 3>`
4. :ref:`distributed in a cluster environment<example 4>`

| Each example, showcases how to setup HpBandSter in different environments and highlights specifics for it.
  Every compute environment is slightly different, but it should be easy to bootstrap from one of the examples and adapt it to any specific needs.
  The first example slowly introduces the main workflow for any HpBandSter run.
  the following ones gradually add complexity by including more features.


.. _example 1:

1. A Local and Sequential Run
+++++++++++++++++++++++++++++

| We are now ready to look at our first real :doc:`example <auto_examples/example_1_local_sequential>` to illustrate how HpBandSter is used.
  Every run consists of the same 5 basic steps which we will now cover.

Step 1: Start a :py:class:`Nameserver <hpbandster.core.nameserver.NameServer>` 

| To initiate the communication between the worker(s) and the optimizer, HpBandSter requires a nameserver to be present.
  This is a small service that keeps track of all running processes and their IP addresses and ports.
  It is a building block that HpBandster inherits from `Pyro4 <https://pythonhosted.org/Pyro4/nameserver.html>`_.
  In this first example, we will run it using the loop back interface with the IP `127.0.0.1`.
  Using the `port=None` argument, will make it use the default port 9090.
  The `run_id` is used to identify individual runs and needs to be given to all other components as well (see below).
  For now, we just fix it to `example1`.

  .. literalinclude:: ../../hpbandster/examples/example_1_local_sequential.py
    :lines: 31-32

Step 2: Start a :py:class:`Worker <hpbandster.core.worker.Worker>` 

| The worker implements the actual problem that is optimized.
  By deriving your worker from the  :py:class:`base worker <hpbandster.core.worker.Worker>` and implementing the `compute` method, it can easily be instantiated with all arguments your specific `__init__` requires and the additional arguments from the base class. The bare minimum is the location of the nameserver and the `run_id`.

  .. literalinclude:: ../../hpbandster/examples/example_1_local_sequential.py
    :lines: 39-40

Step 3: Run an :py:class:`Optimizer <hpbandster.core.master.Master>`

| The optimizer decides which configurations are evaluated, and how the budgets are distributed.
  Besides :py:class:`Random Search <hpbandster.optimizers.randomsearch.RandomSearch>`, and :py:class:`HyperBand <hpbandster.optimizers.hyperband.HyperBand>`, there is :py:class:`BOHB <hpbandster.optimizers.bohb.BOHB>` our own combination of Hyperband and Bayesian Optimization that we will use here.
  Checkout out the :doc:`list of available optimizers <optimizers>` for more info.

| At least, we have to provide the description of the search space, the `run_id`, the nameserver and the budgets.
  The optimization starts when the `run` method is called with the number of iterations as the only mandatory argument.

  .. literalinclude:: ../../hpbandster/examples/example_1_local_sequential.py
    :lines: 46-50

Step 4: Stop all services

| After the run is finished, the services started above need to be shutdown.
  This ensures that the worker, the nameserver and the master all properly exit and no (daemon) threads keep running afterwards.
  In particular we shutdown the optimizer (which shuts down all workers) and the nameserver.

  .. literalinclude:: ../../hpbandster/examples/example_1_local_sequential.py
    :lines: 54-55


Step 5: Analysis of the Results

| After a run is finished, one might be interested in all kinds of information.
  HpBandSter offers full access to all evaluated configurations including timing information and potential error messages for failed runs.
  In this first example, we simply look up the best configuration (called incumbent), count the number of configurations and evaluations, and the total budget spent.
  For more details, see some of the other examples and the documentation of the :py:class:`Result <hpbandster.core.result.Result>` class.

.. literalinclude:: ../../hpbandster/examples/example_1_local_sequential.py
    :lines: 62-68

| The complete source code for this example can be found :doc:`here <auto_examples/example_1_local_sequential>`


.. _example 2:

2. A Local Parallel Run using Threads
+++++++++++++++++++++++++++++++++++++

| Let us now extend this example to start multiple workers, each in a separate thread.
  This is a useful mode to exploit a multicore CPU system, if the individual workers get around Python's global interpreter lock.
  For example, many scikit learn algorithms outsource the heavy duty computations to some C module, making them run truly in parallel even if threaded.

| Below, we can instantiate the specified number of workers.
  To emphasize the effect, we introduce a sleep_interval of one second, which makes every function evaluation take a bit of time.
  Note the additional id argument that helps separating the individual workers.
  This is necessary because every worker uses its processes ID which is the same for all threads here.

.. literalinclude:: ../../hpbandster/examples/example_2_local_parallel_threads.py
    :lines: 38-42


| When starting the optimizer, we can add the min_n_workers argument to the run methods to make the optimizer wait for all workers to start.
  This is not mandatory, and workers can be added at any time, but if the timing of the run is essential, this can be used to synchronize all workers right at the start.

.. literalinclude:: ../../hpbandster/examples/example_2_local_parallel_threads.py
    :lines: 54

| The source code can be found here :doc:`here <auto_examples/example_2_local_parallel_threads>`
  Try running it with different number of workers by changing the `--n_worker` command line argument.


.. _example 3:

3. A Local Parallel Run using Different Processes
+++++++++++++++++++++++++++++++++++++++++++++++++

| Before we can go to a distributed system, we shall first extend our toy example to run in different processes.
  In order to do that, we add the `--worker` flag

.. literalinclude:: ../../hpbandster/examples/example_3_local_parallel_processes.py
    :lines: 24

| which will allow us to run the same script for dedicated workers.
  Those only have to instantiate the worker class and call its run method, but this time the worker runs in the foreground.
  After they processed all the configurations and get the shutdown signal from the master the workers simply exit.

.. literalinclude:: ../../hpbandster/examples/example_3_local_parallel_processes.py
    :lines: 30-33

| You can download the source code here :doc:`here <auto_examples/example_3_local_parallel_processes>`
  Try running the script in three different shells, twice with the `--worker` flag.
  To see what is happening, the logging level for this script is set to *INFO*, so messages from the optimizer and the workers are shown.


.. _example 4:

4. A Distributed Run on a Cluster with a Shared File System
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

| Example 3 is already close to the setup for a distributed environment.
  The only things missing are providing a unique run id, looking up the hostname and distributing the nameserver information across all processes.
  So far, the run id was always hard coded, and the nameserver was running on `localhost` (127.0.0.1, which was also the hostname) on the default port.
  We now have to tell all processes which Network Interface Card (NIC) to use and where the nameserver is located.
  To that end, we introduce three new command line arguments:

.. literalinclude:: ../../hpbandster/examples/example_4_cluster.py
    :lines: 72-74

| The first two are self-explanatory, and we will use a shared directory to distribute the nameserver information to every worker.

.. note::
    This is not the only way to distribute this information, but in our experience almost all clusters offer a shared file system accessible by every compute node.
    We have therefore implemented an easy solution for this scenario.
    If that does not cover your use case, you must find another way to distribute the information about the nameserver to all workers.
    It might be an option then to start a static nameserver, for example on the submission node of the cluster.
    That way, you can hard code the information into the script.

| To find a valid host name we can use the convenience function :py:func:`nic_to_host <hpbandster.core.nameserver.nic_name_to_host>` which looks up a valid hostname for a given NIC.


.. literalinclude:: ../../hpbandster/examples/example_4_cluster.py
    :lines: 80
 

| When creating the nameserver, we can provide the `working_directory` argument to make it store its hostname and port upon start.
  Both values are also returned by the `start` method so that we can use them in the master directly.
  
.. literalinclude:: ../../hpbandster/examples/example_4_cluster.py
    :lines: 92-93
 
| The workers can then simply retrieve that information by loading it from disc:

.. literalinclude:: ../../hpbandster/examples/example_4_cluster.py
    :lines: 83-88

| For the master, we can usually afford to run a worker in the background, as most optimizers have very little overhead.

.. literalinclude:: ../../hpbandster/examples/example_4_cluster.py
    :lines: 98-99

| We also provide the `host`, `nameserver`, and `nameserver_port` arguments to the optimizer.
  Once the run is done, we usually do not want to print out any information, but rather store the result for later analysis.
  Pickling the object returned by the optimizer's run is a very easy way of doing that.

.. literalinclude:: ../../hpbandster/examples/example_4_cluster.py
    :lines: 115-116

| The full example can be found :doc:`here <auto_examples/example_4_cluster>`.
  There you will also find an example shell script to submit the program on a cluster
  running the Sun Grid Engine.





What to read next
~~~~~~~~~~~~~~~~~

| If you are now excited about trying HpBandSter on your problem, you might want to consider reading the :doc:`best practices<best_practices>` on how to make sure you do not run into problems when writing your worker and running the optimizer.
  The :doc:`gallery <auto_examples/index>` also contains some useful scripts to start from, especially workers training PyTorch, and Keras models.

| If you run into problems, please check the FAQ and the Github issues first, before contacting us.
  


