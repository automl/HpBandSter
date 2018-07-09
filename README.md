# HpBandSter
a distributed Hyperband implementation on Steroids

## How to install

A simple
```
pip install hpbandster
```
should do. It should install the dependencies automatically.


If you want to develop on the code you could install it via

```
python3 setup.py develop --user
```


## How to use

Right now, there are only three example showing how to use. You can find them in the `hpbandster/examples` folder.
The basic components of every run are
1. a `NameServer` keeping track of the master's and workers' IPs and ports
2. a `Master` (Usually `BOHB` or `HyperBand`) that coordinates the work
3. one or more `Worker` instances that perform the actual computations.

The first example shows how to do computations locally on one machine with all workers launched in separate threads.
The second one is closer to a usecase on a cluster (SGE in this case).
It contains launching a nameserver and distributing its credentials over a shared filesystem (often found on clusters),
starting a master and multiple workers. The workers don't really perform anything interesting, but hopefully it can
illustrated the usage.
The third example shows how live results can be logged. Use this for long and expensive runs to inspect intermediate results.

## Documentation

There is no real documentation, although many methods and functions do have docstrings.
There is going to be a Sphinx at some point with a bit more detail.
