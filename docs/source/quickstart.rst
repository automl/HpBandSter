Quickstart Guide
================

.. contents::
   :local:

How to use HpBandster
~~~~~~~~~~~~~~~~~~~~~

To get started, we will guide you through a some basic examples.

- First, we will explain the basic usage of BOHB by optimizing a toy function with
- Second, we will expand the optimizer to parallel evaluations with a more realistic example
- Third, we will show you how to do warmstarting with BOHB and how to use the visualization tool
- Also, you can use BOHB with `CAVE <https://github.com/automl/CAVE>`_ to analyze the results



1st example - local, sequential evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the first example, you'll see the basic usage of BOHB. 
It'll run locally with a simple setup. 
To use BOHB, basically you need to follow the next three steps:

1) Set up a nameserver
2) Implement a worker and start it
3) Initialize the BOHB-Optimizer-object and run it.  

.. literalinclude:: /examples/example_1_simple_locally/example1.py

The Implementation of the worker with a simple compute function, which was used above

.. literalinclude:: /examples/commons.py
	:language: python
	:lines: 7-8, 10-43

2nd example - local, parallel evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The purpose of this example is to show

1) how to use multiple workers in parallel
2) how to log live results
3) how a more complex realistic worker could look like

The worker is a simple implementation of a RNN on the 20 Newsgroups dataset.
The results are not really good, but a RNN on character level and with
all the the simple features ('header', 'footer' and 'quotes') removed,
this is actually a hard problem, especially if no word embeddings are
used.

.. literalinclude:: /examples/example_5_rnn_20_newsgroups/run_me.py

Following a extract from the worker code:

The can find the complete example in hpbandster/examples/example_5_rnn_20_newsgroups

.. literalinclude:: /examples/example_5_rnn_20_newsgroups/worker.py
    :lines: 9-11, 18-22, 39-178

3rd example - Warmstarting
~~~~~~~~~~~~~~~~~~~~~~~~~~

This examples covers the warmstart functionality.
We will start a optimizer run with a small budget.
Then we'll shutdown the master, but keep the nameserver alive.
And finally, restarting the optimization run with a new master and a larger budget.

In the end,  we'll introduce a interactive visualization tool.
With this tool, we can illustrate the progress the optimizer made.

.. literalinclude:: /examples/example_4_warmstarting_visualization/run_me.py
    :lines: 20-129

BOHB with CAVE
~~~~~~~~~~~~~~

To run CAVE on BOHB-results, you need a folder with the files *results.json*, *configs.json* and *configspace.pcs* in
it. CAVE will output an individual report for each budget. Simply run

    .. code-block:: bash

        cave --folder bohb_results_folder --ta_exec_dir folder_from_which_bohb_was_run --file_format BOHB

.. note::

        To use CAVE with BOHB, currently you will have to install CAVE from the development-branch
        (e.g. `pip install git+https://github.com/automl/CAVE@development`).
