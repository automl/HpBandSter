import time
import numpy as np

from hpbandster.distributed.worker import Worker
from hpbandster.config_generators.lcnet import LCNetWrapper
import hpbandster.distributed.utils


import ConfigSpace as CS

import logging

logging.basicConfig(level=logging.INFO)


class ToyExample(Worker):
    def __init__(self, min_budget, max_budget, **kwargs):
        self.min_budget = min_budget
        self.max_budget = max_budget
        super(ToyExample, self).__init__(**kwargs)

    def compute(self, config, budget, *args, **kwargs):
        time.sleep(40)

        def toy_example(t, a, b):
            return (10 + a * np.log(b * t)) / 10. + 10e-3 * np.random.rand()

        lc = [1 - toy_example(t / self.max_budget, config["a"], config["b"]) for t in
              np.linspace(min_budget, int(budget), int(budget) / min_budget)]

        return ({
            'loss': lc[-1],
            'info': {"learning_curve": lc}
        })


nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()

min_budget = 1
max_budget = 100

w = ToyExample(min_budget=min_budget, max_budget=max_budget, nameserver=nameserver, ns_port=ns_port)
w.run(background=True)

config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('a', lower=0, upper=1))
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('b', lower=0, upper=1))
CG = LCNetWrapper(config_space=config_space, max_budget=max_budget)

HB = hpbandster.HB_master.HpBandSter(
    config_generator=CG,
    run_id='0',
    eta=2, min_budget=min_budget, max_budget=max_budget,  # HB parameters
    nameserver=nameserver,
    ns_port=ns_port,
    job_queue_sizes=(0, 1),
)
res = HB.run(1, min_n_workers=1)


HB.shutdown(shutdown_workers=True)

print(res.get_incumbent_trajectory())
