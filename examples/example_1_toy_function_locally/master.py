import sys
sys.path.append('../..')
sys.path.append('/ihome/sfalkner/repositories/github/ConfigSpace/')


import hpbandster
import ConfigSpace as CS

import logging
logging.basicConfig(level=logging.DEBUG)


config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
CG = hpbandster.config_generators.RandomSampling(config_space)



HB = hpbandster.HB_master.HpBandSter(
				config_generator = CG,
				run_id = '0',							# this needs to be unique for concurent runs, i.e. when multiple
														# instances run at the same time, they have to have different ids
														# I would suggest using the clusters jobID
														
                eta=2,min_budget=1, max_budget=64,      # HB parameters
				nameserver='localhost',
				ping_interval=4,
				job_queue_sizes=(0,1),
				dynamic_queue_size=True,
				)

res = HB.run(1, min_n_workers=1)
HB.shutdown(shutdown_workers=True)

print(res)


