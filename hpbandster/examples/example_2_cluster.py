"""
Example 2 - Run HpBandSter on a cluster:
========================================

This example shows how to run HpBandSter on a cluster.
We assume here that the resource manager is SGE (Sun Grid Engine),
but the example is easily adapted to a different environment
"""

import os
import argparse
import pickle

import logging
logging.basicConfig(level=logging.DEBUG)

import ConfigSpace as CS
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
from hpbandster.examples.commons import MyWorker


# Parse the command line arguments, which will be passed by the shell script
parser = argparse.ArgumentParser(description='HpBandSter example 2.')
parser.add_argument('--run_id', help='unique id to identify the HPB run.', default='HPB_example_2', type=str)
parser.add_argument('--array_id', help='SGE array id to tread one job array as a HPB run.', default=1, type=int)
parser.add_argument('--working_dir', help='working directory to store live data.', default='.', type=str)
parser.add_argument('--nic_name', help='name of the Network Interface Card.', default='lo', type=str)
args = parser.parse_args()

# Create a config space for the worker
config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))

# One the first node, we start the nameserver, and the master.
# BOHB is usually so cheap, that we can affort to run a worker on the master node, too.
if args.array_id == 1:
    # start nameserver
    NS = hpns.NameServer(run_id=args.run_id, nic_name=args.nic_name,
                         working_directory=args.working_dir)
    ns_host, ns_port = NS.start()  # stores information for workers to find in working_directory

    # Start a worker and pass the nameserver credentials.
    worker = MyWorker(nameserver=ns_host, nameserver_port=ns_port, run_id=args.run_id)
    worker.run(background=True)

    # Initialise the master (here: BOHB)
    HB = BOHB(configspace=config_space,
              run_id=args.run_id,
              eta=3, min_budget=27, max_budget=243,
              host=ns_host,
              nameserver=ns_host,
              nameserver_port=ns_port,
              ping_interval=3600,
              )

    # And start the master for 4 iterations.
    # By passing the argument min_n_workers = 4, the master will wait
    # until at least four workers are online before starting
    res = HB.run(n_iterations=4,
                 min_n_workers=4)

    # pickle result here for later analysis
    with open(os.path.join(args.working_dir, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    # shutdown all workers
    HB.shutdown(shutdown_workers=True)

    # and the nameserver
    NS.shutdown()

else:
    # Translates the name of a network card into a valid host name
    host = hpns.nic_name_to_host(args.nic_name)

    # workers only instantiate the MyWorker, find the nameserver and start serving
    w = MyWorker(run_id=args.run_id, host=host)
    w.load_nameserver_credentials(args.working_dir)

    # run worker in the forground,
    w.run(background=False)
