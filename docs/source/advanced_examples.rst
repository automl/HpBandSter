Advanced examples
=================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:

.. _3rd example:

1st example - Continue Runs and Visualize Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example we will cover the functionalities:

| :ref:`live logging`
| :ref:`warmstart`
| :ref:`warmstart visualization`
| :ref:`BOHB with CAVE`

The full example can be found :doc:`here <auto_examples/example_3_warmstarting_visualization>`

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

2nd example - Combine BOHB and CAVE to analyze results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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