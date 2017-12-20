# HpBandSter
a distributed Hyperband implementation on Steroids


##TODOs

- fix/adjust/add unit tests
- add example showing how to use the visualizer
- add reentrance for jobs on unresponsive hosts
- HMAC key for the nameserver and the dispatcher/master/worker
- add option for workers to die immediately if no dispatcher is available
- add idle time for workers before shutting down (to release workers early)
- add mechanism to determine maximum number of parallel jobs to release workers early

## How to install

A simple
```
python setup.py install --user
```
should do. It should install the dependencies automatically.


If you want to develop on the code you could install it via

```
python setup.py develop --user
```


## How to use

Right now, there is only one example showing how to use it locally on one machine (`examples/example_1_toy_function_locally`).
Please consult the files in there to see how it is used. In particular, you want to checkout `worker.py` as it contains the toy function,
and maybe `all_in_one_parallel.py` which runs multiple workers simultaniously.
