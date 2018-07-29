"""
Example 6 - RNN 20 - with live logging
======================================

"""

import logging
logging.basicConfig(level=logging.DEBUG)

import hpbandster.core.result as hpres
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers.bohb import BOHB

import ConfigSpace as CS
from ConfigSpace.read_and_write import json, pcs_new

from example_6_rnn_20_newsgroups_worker import RNN20NGWorker as MyWorker

config_space = MyWorker.get_config_space()

# This example shows how to log live results. This is most useful
# for really long runs, where intermediate results could already be
# interesting. The results submodule contains the functionality to
# read the two generated files (results.json and configs.json) and
# create a Result object. See below!
# Specify the directory and whether or not existing files are overwritten
result_logger = hpres.json_result_logger(directory='results_example_rnn', overwrite=True)
with open('results_example_rnn/configspace.pcs', 'w') as fh:
    fh.write(pcs_new.write(config_space))


# Every run has to have a unique (at runtime) id.
# This needs to be unique for concurrent runs, i.e. when multiple
# instances run at the same time, they have to have different ids
run_id = '0'


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
num_workers = 1
workers=[]
for i in range(num_workers):
    w = MyWorker(   nameserver=ns_host, nameserver_port=ns_port,
                    run_id=run_id,   # unique Hyperband run id
                    id=i             # unique ID as all workers belong to the same process
                )
    w.run(background=True)
    workers.append(w)


# Step 3:
# In the last of the 3 Steps, we create a optimizer object.
# It samples configurations from the ConfigurationSpace.
# The number of sampled configurations is determined by the
# parameters eta, min_budget and max_budget.
# After evaluating each configuration, starting with the minimum budget
# on the same subset size, only a fraction of 1 / eta of them
# 'advances' to the next round. At the same time the current budget will be doubled.
# This process runs until the maximum budget is reached.
HB = BOHB(  configspace = config_space,
            run_id = run_id,
            eta=3, min_budget=9, max_budget=243,     # HB parameters
            nameserver=ns_host,
            nameserver_port = ns_port,
            result_logger=result_logger,
            ping_interval=10**6
          )

# Then start the optimizer. The n_iterations parameter specifies
# the number of iterations to be performed in this run
# It will wait till minimum n workers are ready
HB.run(n_iterations=4, min_n_workers=num_workers)

# After the optimizer run, we shutdown the master.
HB.shutdown(shutdown_workers=True)
NS.shutdown()

# Just to demonstrate, let's read in the logged runs rather than the returned result from HB.run
res = hpres.logged_results_to_HB_result('.')

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
