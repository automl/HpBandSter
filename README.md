# HpBandSter [![Build Status](https://travis-ci.org/automl/HpBandSter.svg?branch=master)](https://travis-ci.org/automl/HpBandSter)  [![codecov](https://codecov.io/gh/automl/HpBandSter/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/HpBandSter)
a distributed Hyperband implementation on Steroids

## News: Not Maintained Anymore!

Please note that we don't maintain this repository anymore. We also cannot ensure that we can reply to issues in the issue tracker or look into PRs. 

We offer two successor  packages which showed in our [HPOBench paper](https://arxiv.org/abs/2109.06716) superior performance:

1. [SMAC3](https://github.com/automl/SMAC3): is a versatile HPO package with different HPO strategies. It also implements the main idea of BOHB, but uses a RF (or GP) as a predictive model instead of a KDE.
2. [DEHB](https://github.com/automl/dehb): is a HPO package using a combination of differential evolution and hyperband. 

In particular, SMAC3 has an active group of developers working on it and maintaining it. So, we strongly recommend using one of these two packages instead of HPBandSter.

## Overview

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
