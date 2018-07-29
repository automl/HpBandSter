"""
Example 5: MNIST
================
"""

import logging
logging.basicConfig(level=logging.DEBUG)

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from example_5_mnist_worker import MyWorker

import time

run_id = '0'

# This example shows how to log live results. This is most useful
# for really long runs, where intermediate results could already be
# interesting. The results submodule contains the functionality to
# read the two generated files (results.json and configs.json) and
# create a Result object. See below!
# Specify the directory and whether or not existing files are overwritten
result_logger = hpres.json_result_logger(directory='.', overwrite=True)


NS = hpns.NameServer(run_id=run_id, host='localhost', port=0)
ns_host, ns_port = NS.start()


# Initialize a given number of workers
num_workers = 4
workers = []
for i in range(num_workers):
    w = MyWorker(   nameserver=ns_host,
                    nameserver_port=ns_port,
                    run_id=run_id,  # unique Hyperband run id
                    id=i
                )
    w.run(background=True)
    workers.append(w)



HB = BOHB(  configspace=MyWorker.get_configspace(),
            run_id=run_id,
            eta=3, min_budget=1, max_budget=25,  # Hyperband parameters
            nameserver=ns_host,
            nameserver_port=ns_port,
            result_logger=result_logger,
            ping_interval=10**6
         )

res = HB.run(n_iterations=2, min_n_workers=num_workers)

HB.shutdown(shutdown_workers=True)
NS.shutdown()

id2config = res.get_id2config_mapping()

print('A total of %i unique configurations where sampled.'%len(id2config.keys()))
print('A total of %i runs where executed.'%len(res.get_all_runs()))


incumbent_trajectory = res.get_incumbent_trajectory()


import matplotlib.pyplot as plt
plt.plot(incumbent_trajectory['times_finished'], incumbent_trajectory['losses'])
plt.xlabel('wall clock time [s]')
plt.ylabel('incumbent loss')
plt.show()
