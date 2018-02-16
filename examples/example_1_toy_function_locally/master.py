import logging
logging.basicConfig(level=logging.INFO)



from hpbandster.api.optimizers.hyperband import HyperBand
import hpbandster.api.util as hputil

import ConfigSpace as CS


from worker import MyWorker




config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))


ns_host, ns_port = hputil.start_local_nameserver()


w1 = MyWorker(run_id='0', nameserver=ns_host, nameserver_port=ns_port, id='0')
w2 = MyWorker(run_id='0', nameserver=ns_host, nameserver_port=ns_port, id='1')



w1.run(background=True)
w2.run(background=True)


#racing condition of w2 starting after HB found w1 and doesn't ping

HB = HyperBand(
				config_space = config_space,
				run_id = '0',							# this needs to be unique for concurent runs, i.e. when multiple
														# instances run at the same time, they have to have different ids
														# I would suggest using the clusters jobID
														
                eta=2,min_budget=1, max_budget=4,      # HB parameters
				nameserver=ns_host,
				nameserver_port = ns_port,
				ping_interval=3600,
				
				)



HB.wait_for_workers(min_n_workers=2)

res = HB.run(2)
HB.shutdown(shutdown_workers=True)

print(res)


