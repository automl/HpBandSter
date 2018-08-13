"""
Example 8 - Warmstarting for MNIST
==================================

Sometimes it is desired to continue an already finished run because the optimization
requires more function evaluations. In other cases, one might wish to use results
from previous runs to speed up the optimization. This might be useful if initial
runs were done with relatively small budgets, or on only a subset of the data to
get an initial understanding of the problem.

Here we shall see how to use the results from example 5 to initialize BOHB's model.
What changed are
- the number of training points is increased from 8192 to 32768
- the number of validation points is increased from 1024 to 16384
- the mimum budget is now 3 instead of 1 because we have already quite a few runs for a small number of epochs

Note that the loaded runs will show up in the results of the new run. They are all
combined into an iteration with the index -1 and their time stamps are manipulated
such that the last run finishes at time 0 with all other times being negative.
That info can be used to filter those runs when analysing the run.

"""
import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB

import logging
logging.basicConfig(level=logging.DEBUG)



parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum number of epochs for training.',    default=3)
parser.add_argument('--max_budget',   type=float, help='Maximum number of epochs for training.',    default=9)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=4)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
parser.add_argument('--nic_name',type=str, help='Which network interface to use for communication.', default='lo')
parser.add_argument('--shared_directory',type=str, help='A directory that is accessible for all processes, e.g. a NFS share.', default='.')
parser.add_argument('--backend',help='Toggles which worker is used. Choose between a pytorch and a keras implementation.', choices=['pytorch', 'keras'], default='keras')
parser.add_argument('--previous_run_dir',type=str, help='A directory that contains a config.json and results.json for the same configuration space.', default='./example_5_run/')

args=parser.parse_args()


if args.backend == 'pytorch':
	from example_5_pytorch_worker import PyTorchWorker as worker
else:
	from example_5_keras_worker import KerasWorker as worker


# Every process has to lookup the hostname
host = hpns.nic_name_to_host(args.nic_name)


if args.worker:
	import time
	time.sleep(5)	# short artificial delay to make sure the nameserver is already running
	w = worker(run_id=args.run_id, host=host, timeout=120)
	w.load_nameserver_credentials(working_directory=args.shared_directory)
	w.run(background=False)
	exit(0)


# This example shows how to log live results. This is most useful
# for really long runs, where intermediate results could already be
# interesting. The core.result submodule contains the functionality to
# read the two generated files (results.json and configs.json) and
# create a Result object.
result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=False)


# Start a nameserver:
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Start local worker
w = worker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120)
w.run(background=True)


# Let us load the old run now to use its results to warmstart a new run with slightly
# different budgets in terms of datapoints and epochs.
# Note that the search space has to be identical though!
previous_run = hpres.logged_results_to_HBS_result(args.previous_run_dir)


# Run an optimizer
bohb = BOHB(  configspace = worker.get_configspace(),
			  run_id = args.run_id,
			  host=host,
			  nameserver=ns_host,
			  nameserver_port=ns_port,
			  result_logger=result_logger,
			  min_budget=args.min_budget, max_budget=args.max_budget, 
			  previous_result = previous_run,				# this is how you tell any optimizer about previous runs
		   )
res = bohb.run(n_iterations=args.n_iterations)

# store results
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
	pickle.dump(res, fh)

# shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

