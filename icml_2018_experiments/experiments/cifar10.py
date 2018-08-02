import os
import pickle
import argparse

import numpy as np

from workers.cifar10 import CIFAR10_SSCO as Worker
import util

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB
from hpbandster.core.result import json_result_logger
# deactivate debug output for faster experiments
import logging
logging.basicConfig(level=logging.DEBUG)


################################################################################
#                    Benchmark specific stuff
################################################################################
parser = argparse.ArgumentParser(description='Run different optimizers to optimize BNNs on different datasets.', conflict_handler='resolve')
parser = util.standard_parser_args(parser)


# add benchmark specific arguments
parser.add_argument('--min_budget', type=int, help='Minimum number of epochs to train.', default=1)
parser.add_argument('--max_budget', type=int, help='Maximum number of epochs to train.', default=3)
parser.add_argument('--num_GPUs', type=int, help='Number of GPUs used per worker.', default=1)
parser.add_argument('--method', type=str, choices=['bohb'], default='bohb')
parser.add_argument('--torch_source', type=str, help='Pointer to the directory with the torch source code. Default (None) will use the sources under "./workers/lib/cifar10_cutout_validation" .', default=None)
parser.add_argument('--array_id', type=int, help='Unique ID for each process with the same run ID. The process with the ID 1 will start a master and a worker, while all greater IDs will only start a worker.', default=1)



args = parser.parse_args()

args = parser.parse_args('--method bohb'.split())


# every id starts a worker
worker = Worker(run_id=args.run_id, nGPU=args.num_GPUs, torch_source_path=args.torch_source)


if args.array_id == 1:

	# directory where the results are stored
	dest_dir = os.path.join(args.dest_dir, "CIFAR10")
	
	# setup a nameserver
	NS = hpns.NameServer(run_id=args.run_id, nic_name=args.nic_name, working_directory=args.working_directory)
	ns_host, ns_port = NS.start()

	configspace=worker.get_config_space()

	# start worker in the background
	worker.load_nameserver_credentials(args.working_directory)
	worker.run(background=True)
	
	# this experiment takes a while and we want to store all intermediate information
	# as soon as possible, so we log all results in a json file
	res_logger = json_result_logger(dest_dir, overwrite=False)
	
	HPB = BOHB( configspace, working_directory=args.dest_dir,
					run_id = args.run_id,
					min_budget=args.min_budget, max_budget=args.max_budget,
					host=ns_host,
					nameserver=ns_host,
					nameserver_port = ns_port,
					ping_interval=3600,
					result_logger=res_logger,
				)

	result = HPB.run(n_iterations = args.num_iterations) 

	with open(os.path.join(dest_dir, 'bohb_full_run_{}.pkl'.format(args.run_id)), 'wb') as fh:
		pickle.dump(extract_results_to_pickle(result), fh)

	# shutdown the worker and the dispatcher
	HPB.shutdown(shutdown_workers=True)
	NS.shutdown()

else:
	# start worker in the background
	worker.load_nameserver_credentials(args.working_directory)
	worker.run(background=False)
