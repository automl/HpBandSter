Frequently Asked Questions
==========================

.. rubric:: I found an issue. What do I do?

If you think you found an issue or a bug, please use our `GitHub issue tracker <https://github.com/automl/HpBandSter/issues>`_.
Provide as much information as you can about your environment as well as a minimum working example that reproduces the issue.


.. rubric::  It seems like some workers are just waiting for something. But I don't receive an error message. Is this a normal behavior?

No, it's not. It will be fixed soon. It occurs if a value returned by the compute method is not serializable. As a workaround, you can cast it to some Python datatype, e.g. a list.


.. rubric:: Pip doesn't find the HpBandster package.

Please check, if you are using pip3. HpBandster is developed for python3.