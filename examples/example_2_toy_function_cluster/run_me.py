import argparse

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
parser.add_argument('--array_id',    help='SGE array id to tread one array of jobs as a HPB run.', default=1, type=int)
parser.add_argument('--working_dir', help='working directory to store live data.', default=None, type=str)



args=parser.parse_args()



if args.array_id == 1:
	# start nameserver
	NS = hputil.NameServer(run_id=args.run_id, host='localhost', working_directory=args.working_dir)


	ns_host, ns_port = NS.start()	# stores information for workers to find in working_directory

	# BOHB is usually so cheap, that we can affort to run a worker on the master node, too.
	worker = MyWorker(nameserver=ns_host, nameserver_port=ns_port, run_id=args.run_id)
	worker.run(background=True)


	HPB = BOHB(	configspace = config_space,
				run_id = args.run_id,
                eta=3,min_budget=27, max_budget=243,
				nameserver=ns_host,
				nameserver_port = ns_port,
				ping_interval=3600,	
		)
	
	res = HPB.run(	n_iterations = 4,
					min_n_workers = 1		# BOHB can wait until a minimum number of workers is online before starting
		)
	
	# pickle result here for later analysis
	with open(os.path.join(args.working_dir, 'results.pkl', 'wb') as fh:
		pickle.dump(fh, res)
	
	# shutdown all workers
	HPB.shutdown(shutdown_workers=True)
	
	# and the nameserver
	NS.shutdown()

else:

	# workers only instantiate the MyWorker, find the nameserver and start serving
	w = MyWorker(nameserver=ns_host, nameserver_port=ns_port, run_id=args.run_id)
	w.load_nameserver_credentials(args.working_dir)
	# run worker in the forground, 
	w.run(background=False)
