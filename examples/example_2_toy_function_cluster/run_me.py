import argparse

import logging
logging.basicConfig(level=logging.DEBUG)

from hpbandster.api.optimizers.hyperband import HyperBand
import hpbandster.api.util as hputil

import ConfigSpace as CS

from worker import MyWorker

config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))




parser = argparse.ArgumentParser(description='HpBandSter example 2.')
parser.add_argument('--run_id',      help='unique id to identify the HPB run.', default=None, type=str)
parser.add_argument('--array_id',    help='SGE array id to tread one array of jobs as a HPB run.', default=None, type=str)
parser.add_argument('--working_dir', help='working directory to store live data.', default='.', type=str)



args=parser.parse_args()


# resolve network ip by looking up the address of eth0
host = hputil.nic_name_to_host('eth0')





if args.array_id == 1:
	# start nameserver
	ns_host, ns_port = hputil.start_local_nameserver(host=host, port=0)

	# store information for workers to find




else

# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine
# with a random port




# Start a bunch of workers in some threads, just to show how it works.
# On the cluster, each worker would run in a separate job and the nameserver
# credentials have to be distributed.
num_workers = 1

workers=[]
for i in range(num_workers):
	w = MyWorker(	nameserver=ns_host, nameserver_port=ns_port,
					run_id=run_id, # unique Hyperband run id
					id=i	# unique ID as all workers belong to the same process
					)
	w.run(background=True)
	workers.append(w)


HB = HyperBand(	# Try BOHB here instead of Hyperband:
				# simply add
				# from hpbandster.api.optimizers.bohb import BOHB
				configspace = config_space,
				run_id = '0',
                eta=3,min_budget=27, max_budget=243,	# HB parameters
				nameserver=ns_host,
				nameserver_port = ns_port,
				ping_interval=3600,					# here, master pings for workers every hour 
				)

res = HB.run(4, min_n_workers=num_workers)
HB.shutdown(shutdown_workers=True)



id2config = res.get_id2config_mapping()

print('A total of %i unique configurations where sampled.'%len(id2config.keys()))
print('A total of %i runs where executed.'%len(res.get_all_runs()))

incumbent_trajectory = res.get_incumbent_trajectory()

import matplotlib.pyplot as plt
plt.plot(incumbent_trajectory['times_finished'], incumbent_trajectory['losses'])
plt.xlabel('wall clock time [s]')
plt.ylabel('incumbent loss')
plt.show()

