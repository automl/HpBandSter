import sys
sys.path.append('../..')
sys.path.append('/home/sfalkner/repositories/github/ConfigSpace/')


import hpbandster
from hpbandster.distributed.worker import Worker
import hpbandster.distributed.utils
import ConfigSpace as CS

import logging
logging.basicConfig(level=logging.INFO)

# starts a local nameserve
nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()


# import the definition of the worker (could be in here as well, but is imported to reduce code duplication)
from worker import MyWorker

# starting the worker in a separate thread
num_workers=64
workers = []
for n in range(num_workers):
	w = MyWorker(nameserver=nameserver, ns_port=ns_port, id=n)	# unique ID as all workers belong to the same process
	w.run(background=True)
	workers.append(w)

# simple config space here: just one float between 0 and 1
config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
CG = hpbandster.config_generators.RandomSampling(config_space)


# instantiating Hyperband with some minimal configuration
HB = hpbandster.HB_master.HpBandSter(
				config_generator = CG,
				run_id = '0',							# this needs to be unique for concurent runs, i.e. when multiple
														# instances run at the same time, they have to have different ids
														# For this all_in_one example, it doesn't reall matter, as the
														# nameserver is unique to this run
														
                eta=2,min_budget=1, max_budget=64,      # HB parameters
				nameserver=nameserver,
				ns_port = ns_port,
				job_queue_sizes=(0,1),
				dynamic_queue_size=True
				)
#runs 16 iteration if at least num_workers workers are available
res = HB.run(16, min_n_workers=num_workers)

# shutdown the workers and the dispatcher
HB.shutdown(shutdown_workers=True)

print(res.get_incumbent_trajectory(all_budgets=False))


