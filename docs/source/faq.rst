Frequently Asked Questions
==========================

**Q:** It seems like some workers are just waiting for something. But i don't receive a error message. Is this a normal behaviour?

**A:**	No, it's not. It will be fixed soon. It occurs if a value, which is returned by the compute method is not serializable. As a workaround, you can cast it to some python datatype, e.g. a list.



**Q:** Pip doesn't find the HpBandster package.

**A:** Please check, if you are using pip3. HpBandster is developed for python3.