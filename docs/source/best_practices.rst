Best Practices
~~~~~~~~~~~~~~


This page contains some (hopefully) helpful tips on how to use HpBandSter.
It is a mixture of how to go about implementing your worker, picking your budgets, and running the optimizers with the right parameters.

  1. :ref:`implement worker`

  2. :ref:`pick optimizer`


.. _implement worker:

Implement your Worker
++++++++++++++++++++++++

a. Attach the ConfigSpace to the worker
---------------------------------------
As those two are usually tightly interlinked, it only makes sense to include the definition of the search space into the worker.
We usually add a static method called `get_configspace` to it that returns the corresponding ConfigSpace object.
This way you can make sure that you use the same search space in all scripts using this worker.

b. Run your Worker locally with a tiny Budget
---------------------------------------------
This might sound stupid, but we think it is worth pointing out the obvious sometimes.
When implementing your worker, you can simply include the following lines at the end of the script to make it runnable and execute the worker's compute function with a random configuration.

.. literalinclude:: ../../hpbandster/examples/example_5_keras_worker.py
    :lines: 243-250

Simply change the name of the worker class to your class name, and pick a tiny budget.
This will allow you to quickly try a random configuration quickly and catch simple mistakes early.

c. Run a short Run locally with Debug Output
--------------------------------------------
Most of the times, when HpBandSter seems to stall and do nothing, it is due to a minor mistake in either the worker, its return values or the setup.
To eliminate the first two, change :doc:`example 1 <auto_examples/example_1_local_sequential>` to use your worker and run it with tiny budgets for one iteration.
By setting the logging level to ``DEBUG`` you will see every exception thrown on the worker side that might get lost otherwise.

Once this is working, you can bootstrap from the other examples to run HpBandSter in the suitable setting.
The examples can serve as a starting point, but are, by no means, the only way to set things up.
They are meant to alleviate the burden of some boilerplate code necessary due to the flexibility of HpBandSter.


.. _pick optimizer:

Pick and Configure the optimizer
+++++++++++++++++++++++++++++++++++

Once you have your worker implemented it is time to think about the actual optimization run.
There are several things to consider:

a. How many iterations should I use?
------------------------------------
The short answer is: As much as you can afford.

But reality is a bit more nuanced than that. It depends on which optimizer you are using, how many workers you have and how much resources you are willing to spend.

For optimizers that rely on random sampling, there is no interaction between the number of workers and the number of iterations you do, but for the model based ones there are some things to keep in mind.
First, the more workers you have, the more concurrent runs there are. This means that initially (when there is no model) more random configurations are drawn.
Consequently, the model is potentially build later if more workers are in the pool, which could negatively impact the optimization especially for few iterations.
Second, the more workers are present, the more queries to the model are made before new results come in.
This might lead to the procedure being stuck in a local optimum for a longer period compared to a sequential run.

b. How do I pick the budgets?
-----------------------------
That is a tough question to answer in general, and completely problem dependent.
What we usually do when applying BOHB to a new problem is the following:

  1. We simplify the config space if necessary/possible, to be at most 10 dimensional to make the problem easier.
  2. We do a short optimization run on what we think are `reasonable` budgets. The `max_budget` does not have to be the one we actually aim for in the end, but should be big enough that we are confident it is representative of the full problem.
  3. We analyze the run, in particular we look at the rank correlations across budgets. If those are really small (<0.2, or so) from one budget to the next bigger one we don't expect Hyperband or BOHB to perform very well.
     In that case, we either increase the minimum budget, or we lower the `eta` parameter, which will shrink the steps between the budgets.
  4. We repeat steps 2 and 3 until we find the optimization to progress satisfactorily.

Ideally, the optimizer would adjust the budgets online, but none of the implement ones does that right now.
We are working on extensions that automate this procedure and incorporate that into the optimization run itself.

c. When should I change the parameters of the optimizer?
--------------------------------------------------------
Hopefully never, but that is probably not true. We know of cases where we had to tweak the default parameters of BOHB to achieve the best performance.
It's all a bit problem depend, so there is no general advice, but a few rules of thumb

If BOHB does not find better solutions for a while, but you know/expect better performance is possible, you can try to make it explore more by
  1. decreasing ``num_samples``
  2. increasing ``top_n_percent``
  3. increasing ``min_bandwidth`` and/or ``bandwidth_factor``








