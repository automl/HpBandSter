import logging
logging.basicConfig(level=logging.DEBUG)

from hpbandster.api.optimizers.hyperband import HyperBand
import hpbandster.api.util as hputil

import ConfigSpace as CS

from worker import MyWorker

config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))


# Every run has to have a unique (at runtime) id.
# This needs to be unique for concurent runs, i.e. when multiple
# instances run at the same time, they have to have different ids
# Here we pick '0'
run_id = '0'




# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine
# with a random port
ns_host, ns_port = hputil.start_local_nameserver(host='localhost', port=0)



# Start a bunch of workers in some threads, just to show how it works.
# On the cluster, each worker would run in a separate job and the nameserver
# credentials have to be distributed.
num_workers = 2

workers=[]
for i in range(num_workers):
	w = MyWorker(	nameserver=ns_host, nameserver_port=ns_port,
					run_id=run_id, # unique Hyperband run id
					id=i	# unique ID as all workers belong to the same process
					)
	w.run(background=True)
	workers.append(w)


HB = HyperBand(
				config_space = config_space,
				run_id = '0',
                eta=3,min_budget=1, max_budget=9,	# HB parameters
				nameserver=ns_host,
				nameserver_port = ns_port,
				ping_interval=3600,					# here, master pings for workers every hour 
				)

res = HB.run(2, min_n_workers=num_workers)
HB.shutdown(shutdown_workers=True)
