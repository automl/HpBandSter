# HpBandSter [![Build Status](https://travis-ci.org/automl/HpBandSter.svg?branch=master)](https://travis-ci.org/automl/HpBandSter)
a distributed Hyperband implementation on Steroids

This python 3 package is a framework for distributed hyperparameter optimization.
It started out as a simple implementation of [Hyperband (Li et al. 2017)](http://jmlr.org/papers/v18/16-558.html), and contains
an implementation of [BOHB (Falkner et al. 2018)](http://proceedings.mlr.press/v80/falkner18a.html)

## How to install

We try to keep the package on PyPI up to date. So you should be able to install it via:
```
pip install hpbandster
```
If you want to develop on the code you could install it via:

```
python3 setup.py develop --user
```

## Documentation

The documentation is hosted on github pages: [https://automl.github.io/HpBandSter/](https://automl.github.io/HpBandSter/)
It contains a quickstart guide with worked out examples to get you started in different circumstances.
Check it out if you are interest in applying one of the implemented optimizers to your problem.

We have also written a [blogpost](https://www.automl.org/blog_bohb/) showcasing the results from our ICML paper.
