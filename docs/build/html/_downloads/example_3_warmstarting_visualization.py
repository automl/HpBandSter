"""
Example 3 - Warmstarting
========================

This examples covers the warmstart functionality
We will start an optimizer run with a small budget
Then we'll shutdown the master, but keep the nameserver alive.
And finally, restarting the optimization run with a new master and a larger budget

In the end, we'll introduce an interactive visualization tool.
With this tool, we can illustrate the progress of the optimizer.
"""

import os
import pickle
import matplotlib.pyplot as plt

import ConfigSpace as CS
import logging
logging.basicConfig(level=logging.DEBUG)

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.examples.commons import MyWorker, sample_configspace
from hpbandster.optimizers import H2BO as opt
from hpbandster.core.result import extract_HB_learning_curves
from hpbandster.visualization import interactive_HB_plot, default_tool_tips


# First, create a ConfigSpace-Object.
# It contains the hyperparameters to be optimized
# For more details, please have a look in the ConfigSpace-Example in the Documentation
config_space = sample_configspace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))

# We use live logging with the jason result logger
result_logger = hpres.json_result_logger(directory='.', overwrite=True)

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
worker = MyWorker(nameserver=ns_host, nameserver_port=ns_port, run_id=run_id)
worker.run(background=True)

# We will start the first run with a smaller budget, which we define here.
# In the second run, we'll use three times as much.
min_budget = 9
max_budget = 243

# Step 3:
# The number of sampled configurations is determined by the
# parameters eta, min_budget and max_budget.
# After evaluating each configuration, starting with the minimum budget
# on the same subset size, only a fraction of 1 / eta of them
# 'advances' to the next round. At the same time the current budget will be doubled.
# This process runs until the maximum budget is reached.
HB = opt(   configspace=config_space,
            run_id=run_id,
            eta=3, min_budget=min_budget, max_budget=max_budget,
            nameserver=ns_host,
            nameserver_port=ns_port,
            result_logger=result_logger,
            min_points_in_model=4,
            min_bandwidth=1e-1
        )


# Do a short run and shutdown the master (but keep the name server and the worker alive)
res = HB.run(2)
HB.shutdown(shutdown_workers=False)


# Now let's start a new run, but feed in the results of the first one to warmstart the model.
# Note that the budgets don't have to align, but beware: if the max budget of the second run is not
# greater or equal to the max budget in the previous runs, BOHB's model might never be updated!
min_budget *= 3
max_budget *= 3

HB = opt(   configspace=config_space,
            run_id=run_id,
            eta=3,min_budget=min_budget, max_budget=max_budget,
            nameserver=ns_host,
            nameserver_port=ns_port,
            previous_result=res,  # Here is where we feed the previous run into our optimizer
            min_points_in_model=4,
            top_n_percent=5,
            bandwidth_factor=1,
            num_samples=32,
            min_bandwidth=1e-1
        )
res = HB.run(4)

# After the optimizer run, we shutdown the master.
HB.shutdown(shutdown_workers=True)
NS.shutdown()


# BOHB will return a result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the result-object, see its documentation.
id2config = res.get_id2config_mapping()
print('A total of %i unique configurations where sampled.'%len(id2config.keys()))
print('A total of %i runs where executed.'%len(res.get_all_runs()))


# Hpbandster contains also a visualization tool to plot the
# 'learning curves' of the sampled configurations
incumbent_trajectory = res.get_incumbent_trajectory()
lcs = res.get_learning_curves(lc_extractor=extract_HB_learning_curves)

tool_tips = default_tool_tips(res, lcs)
fig, ax, check, none_button, all_button = interactive_HB_plot(lcs, tool_tip_strings=tool_tips,
                                                              show=False)
ax.set_ylim([0.1*incumbent_trajectory['losses'][-1], 1])
ax.set_yscale('log')

plt.show()
