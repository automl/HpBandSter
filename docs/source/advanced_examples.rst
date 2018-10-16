Advanced examples
=================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:

.. _3rd example:

Optimizing a small Convolutional Neural Network on MNIST
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example we will cover the functionalities:

| :ref:`complicated configspace`
| :ref:`live logging`
| :ref:`visualization`

The first two parts are covered in :doc:`example 5 <auto_examples/example_5_mnist>` while the visualization is presented in :doc:`example 6 <auto_examples/plot_example_6_analysis>` and  :doc:`example 7 <auto_examples/plot_example_7_interactive_plot>`  

.. _complicated configspace:

1) A more realistic Search Space
++++++++++++++++++++++++++++++++

So far, the search space in the quickstart examples consisted of a single continuous variables.
The interesting problems have, of course, more difficult spaces to search through.
Challenges include

a) higher number of dimensions
b) not only continuous parameters, but also integer values, categorical choices
c) dependencies/conditionalities between parameters

The worker for :doc:`the MNIST example <auto_examples/example_5_mnist>` has a fairly complex search space that covers the challenges above.
Please checkout :doc:`the Keras <auto_examples/example_5_keras_worker>` or  :doc:`the PyTorch <auto_examples/example_5_pytorch_worker>` to see how the search space and how it is encoded.


.. _live logging:

2) Live logging
+++++++++++++++

So far, the results of a run were only available after it finished.
For very long and expensive runs, one might want to check the progress while the optimization is still in progress.
To that end, HpBandSter includes a functionality that we call **live logging**.
The idea is simply to write enough information to disk such that intermediate results can be analysed the same way as completed runs.
The storage is implemented by the :py:class:`result logger <hpbandster.core.result.json_result_logger>` which stores two JSON files in a specified directory.
Loading that information is implemented in :py:func:`this function <hpbandster.core.result.logged_results_to_HBS_result>`


To store the information simply create a :py:class:`result logger <hpbandster.core.result.json_result_logger>` and pass it to an optimizer.
The relevant lines from the :doc:`example <auto_examples/example_5_mnist>` are

.. literalinclude:: ../../hpbandster/examples/example_5_mnist.py
    :lines: 17, 63, 75-82

Please check out :doc:`example <auto_examples/plot_example_6_analysis>` to see how to load these results.

.. _visualization:

3) Visualization
++++++++++++++++

HpBandster contains a few tools to visualize the results of an optimization run.
An example for the static plots can be found in :doc:`example 6 <auto_examples/plot_example_6_analysis>`.
The presented quantities allow one to see how many configurations were evaluated on which budgets, how many concurrent runs were done during the optimization, and how the losses correlated across budgets.

There is also a small interactive plot tool shown in :doc:`example 7 <auto_examples/plot_example_7_interactive_plot>`.
It allows to better explore the performance of the runs by plotting various iterations and allowing to show more information on selected configurations.
It is not the most elaborate tool, but maybe it can help to find patterns in the results.

.. _warmstarting:

Continuing a Run or Warmstarting the Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Under certain circumstances, one might want to continue a run, or reuse old evaluations for a slightly modified objective.
Those include, but are not limited to

- The run ended unexpectedly due to some hardware problems (computer crashed).
- The analysis reveals that the optimization was still improving meaning the run was to short.
- The maximum budget used in the run turned out to be insufficient to achieve the desired performance.
- The finished run might have been an preliminary run on only a subset of the data to speed up the experiment. The obtained results are still good proxies for the full dataset.

In :doc:`example 8 <auto_examples/example_8_mnist_continued>`, we show how one loads the result of an `old run` and initialize the model for a new run.
Specifically, we optimize the same CNN architecture on MNIST, but train on more data points.
We also increased the minimum budget, as we do not expect anymore gains from evaluations after one epoch.

.. note::
   While some specific details of the problem can change, it is important to realize that this only works if the search space is identical, and the old objective is still a very good approximation for the new one!
   Continuing with the same worker, settings, and budgets using this functionality will not recover runs that where pending when the JSON files where written.
   Also, the iterations start counting at zero again, i.e. it might use different budgets as the next iteration of the previous run would have used.
