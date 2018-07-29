"""
Example 4 - How to use the configuration space
==============================================

"""
import logging
logging.basicConfig(level=logging.DEBUG)

import ConfigSpace.read_and_write.json as json_writer

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from example_4_config_space_worker import MyWorker


# Every run has to have a unique (at runtime) id.
# This needs to be unique for concurrent runs, i.e. when multiple
# instances run at the same time, they have to have different ids
run_id = '0'

# We use live logging with the jason result logger
result_logger = hpres.json_result_logger(directory='.', overwrite=True)

# Step 1:
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine
# with a random port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
NS = hpns.NameServer(run_id=run_id, host='localhost', port=0)
ns_host, ns_port = NS.start()

# Step 2:
# The worker implements the connection to the model to be evaluated.
# Its 'compute'-method will be called later by the BOHB-optimizer repeatedly
# with the sampled configurations and return for example the computed loss.
# Further usages of the worker will be covered in a later example.
w = MyWorker(   nameserver=ns_host,
                nameserver_port=ns_port,
                run_id=run_id,  # unique Hyperband run id
            )
w.run(background=True)

# Write the ConfigSpace for later use to the working dir
with open('configspace.json', 'w') as file:
    file.write(json_writer.write(MyWorker.get_configspace()))


# Step 3:
# The number of sampled configurations is determined by the
# parameters eta, min_budget and max_budget.
# After evaluating each configuration, starting with the minimum budget
# on the same subset size, only a fraction of 1 / eta of them
# 'advances' to the next round. At the same time the current budget will be doubled.
# This process runs until the maximum budget is reached.
HB = BOHB(  configspace=MyWorker.get_configspace(),
            run_id=run_id,
            eta=3, min_budget=1, max_budget=25,  # Hyperband parameters
            nameserver=ns_host,
            nameserver_port=ns_port,
            result_logger=result_logger,
            ping_interval=10**6
         )
res = HB.run(n_iterations=2)


# After the optimizer run, we shutdown the master and the nameserver.
HB.shutdown(shutdown_workers=True)
NS.shutdown()


# BOHB will return a result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the result-object, see its documentation.
id2config = res.get_id2config_mapping()
print('A total of %i unique configurations where sampled.'%len(id2config.keys()))
print('A total of %i runs where executed.'%len(res.get_all_runs()))


# The incumbent trajectory is a dictionary with all the configuration IDs, the times the runs
# finished, their respective budgets, and corresponding losses.
# It's used to do meaningful plots of the optimization process.
incumbent_trajectory = res.get_incumbent_trajectory()

import matplotlib.pyplot as plt
plt.plot(incumbent_trajectory['times_finished'], incumbent_trajectory['losses'])
plt.xlabel('wall clock time [s]')
plt.ylabel('incumbent loss')
plt.show()
