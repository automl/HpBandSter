Frequently Asked Questions
==========================

.. rubric:: I found an issue. What do I do?

If you think you found an issue or a bug, please use our `GitHub issue tracker <https://github.com/automl/HpBandSter/issues>`_.
Provide as much information as you can about your environment as well as a minimum working example that reproduces the issue.


.. rubric::  It seems like some workers are just waiting for something. But i don't receive a error message. Is this a normal behaviour?

No, it's not. It will be fixed soon. It occurs if a value, which is returned by the compute method is not serializable. As a workaround, you can cast it to some python datatype, e.g. a list.


.. rubric:: Pip doesn't find the HpBandster package.

Please check, if you are using pip3. HpBandster is developed for python3.