
import hpbandster
from hpbandster.distributed.worker import Worker
import hpbandster.distributed.utils
import ConfigSpace as CS

import logging
logging.basicConfig(level=logging.DEBUG)


# starts a local nameserve
nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()


# import the definition of the worker (could be in here as well, but is imported to reduce code duplication)
from worker import MyWorker


# Every run has to have a unique (at runtime) id.
# This needs to be unique for concurent runs, i.e. when multiple
# instances run at the same time, they have to have different ids
# Here we pick '0'
run_id = '0'


# starting the worker in a separate thread
w = MyWorker(nameserver=nameserver, run_id=run_id, ns_port=ns_port)
w.run(background=True)


# simple config space here: just one float between 0 and 1
config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
CG = hpbandster.config_generators.RandomSampling(config_space)



# instantiating Hyperband with some minimal configuration
HB = hpbandster.HB_master.HpBandSter(
				config_generator = CG,
				run_id = run_id,
                eta=2,min_budget=1, max_budget=64,      # HB parameters
				nameserver=nameserver,
				ns_port = ns_port,
				job_queue_sizes=(0,1),
				)
#runs one iteration if at least one worker is available
res = HB.run(1, min_n_workers=1)

# shutdown the worker and the dispatcher
HB.shutdown(shutdown_workers=True)

print(res.get_incumbent_trajectory())


