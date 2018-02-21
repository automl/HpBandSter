# example to show how to run HpBandSter on a cluster
# We assume here that the resource manager is SGE (Sun Grid Engine),
# but the example is easily adapted to a different environment

import os
import argparse
import pickle

import logging
logging.basicConfig(level=logging.DEBUG)

from hpbandster.api.optimizers.bohb import BOHB
import hpbandster.api.util as hputil

import ConfigSpace as CS

from worker import MyWorker

config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))




parser = argparse.ArgumentParser(description='HpBandSter example 2.')
parser.add_argument('--run_id',      help='unique id to identify the HPB run.', default='HPB_example_2', type=str)
parser.add_argument('--array_id',    help='SGE array id to tread one job array as a HPB run.', default=1, type=int)
parser.add_argument('--working_dir', help='working directory to store live data.', default='.', type=str)


args=parser.parse_args()


if args.array_id == 1:
	# start nameserver
	NS = hputil.NameServer(run_id=args.run_id, nic_name='eth0',
							working_directory=args.working_dir)


	ns_host, ns_port = NS.start()	# stores information for workers to find in working_directory

	# BOHB is usually so cheap, that we can affort to run a worker on the master node, too.
	worker = MyWorker(nameserver=ns_host, nameserver_port=ns_port, run_id=args.run_id)
	worker.run(background=True)


	HPB = BOHB(	configspace = config_space,
				run_id = args.run_id,
                eta=3,min_budget=27, max_budget=243,
                host=ns_host,
				nameserver=ns_host,
				nameserver_port = ns_port,
				ping_interval=3600,	
		)
	
	res = HPB.run(	n_iterations = 4,
					min_n_workers = 4		# BOHB can wait until a minimum number of workers is online before starting
		)
	
	# pickle result here for later analysis
	with open(os.path.join(args.working_dir, 'results.pkl'), 'wb') as fh:
		pickle.dump(res, fh)
	
	# shutdown all workers
	HPB.shutdown(shutdown_workers=True)
	
	# and the nameserver
	NS.shutdown()

else:

	host = hputil.nic_name_to_host('eth0')

	# workers only instantiate the MyWorker, find the nameserver and start serving
	w = MyWorker(run_id=args.run_id, host=host)
	w.load_nameserver_credentials(args.working_dir)
	# run worker in the forground, 
	w.run(background=False)
