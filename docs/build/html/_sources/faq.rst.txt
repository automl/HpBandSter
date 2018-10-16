Frequently Asked Questions
==========================

.. rubric:: I found an issue. What do I do?

If you think you found an issue or a bug, please use our `GitHub issue tracker <https://github.com/automl/HpBandSter/issues>`_.
Provide as much information as you can about your environment as well as a minimum working example that reproduces the issue.


.. rubric::  It seems like some workers are just waiting for something. But I don't receive an error message. Is this a normal behavior?

No, it's not. Please try enabling debug output to see if any exceptions on the worker side occur.
Try running your optimization locally by either using really small budgets, or replacing an expensive computation by a fictitious value.
Often, the problem stems from a value returned by the compute method that is not serializable by the underlying Pyro4 package.
Make sure all values that the compute method returns are build-in Python datatype, e.g. lists and dictionaries.

.. rubric:: Pip doesn't find the HpBandster package.

Please check, if you are using pip3. HpBandster is developed for python3.


.. rubric:: How do I set the budgets and for how many iterations should I optimize?

The meaning of the budgets and the actual values are highly problem specific, so we cannot give a general answer here.
Please check the :doc:`best practices<best_practices>` for some advice on the budgets and the number of iterations.

.. rubric:: None of the configuration sampled by BOHB come from the model?

If you analyse you run and you find that none of the configurations were sampled from the model, there are a few possible explanations:

a. Maybe your run was simply to short. BOHB starts building a model after d+2 observations have been made on any budget (usually first on the smallest one) where d is the number of parameters in the configuration space.
   If your run was relatively short compared to the dimensionality of the space, there might not have been a model to sample from.
   Consider running it longer (check:ref:`this advanced example<warmstarting>` to see how to continue from previous evaluations).
b. Your ConfigurationSpace might contain features not supported by BOHB, namely ordinals and constants. Those are supported by the ConfigSpace package, but are currently not handled by BOHB.
c. If your search space is restricted by several forbidden clauses sampling from the model could yield only forbidden configurations (unlikely in practice, but still possible).
   To make BOHB aware of the constraints and learn to avoid them, try to remove the forbiddens from the configuration space definition and make the worker check for them in the compute.
   If a configuration violates the constraints, you can simply raise an exception in the worker. That way, HpBandSter automatically catches that and model based optimizers will associated the worst possible loss with that configuration.
   That way, BOHB learns to avoid the forbidden regions and tries to actively sample in the space of feasible configurations.


.. rubric:: How can I improve performance in the presence of forbidden clauses?

While the ConfigSpace allows to define forbidden clauses, this way of incorporating constraints is suboptimal for model-based optimizers, because they cannot learn about these constraints. See part c. in the question above for a solution.
