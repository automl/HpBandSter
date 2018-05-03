import os
import pickle

#from hpbandster.optimizers import HyperBand as opt
from hpbandster.optimizers import H2BO as opt
#from hpbandster.optimizers import BOHB as opt



import matplotlib.pyplot as plt

import hpbandster.core.nameserver as hpns
import ConfigSpace as CS

from hpbandster.examples.commons import MyWorker

import logging
logging.basicConfig(level=logging.DEBUG)


config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))


# same basic setup as the previous examples

run_id = '0'
NS = hpns.NameServer(run_id=run_id, host='localhost', port=0)
ns_host, ns_port = NS.start()

worker = MyWorker(	nameserver=ns_host, nameserver_port=ns_port, run_id=run_id)
worker.run(background=True)


min_budget = 9
max_budget = 243


HB = opt(	configspace = config_space,
				run_id = run_id,
                eta=3,min_budget=min_budget, max_budget=max_budget,
				nameserver=ns_host,
				nameserver_port = ns_port,
				min_points_in_model=4,
				min_bandwidth=1e-1
				)

# do a short run and shutdown the master (but keep the name server and the worker alive
res = HB.run(2)
HB.shutdown(shutdown_workers=False)


# now let's start a new run, but feed in the results of the first one to warmstart the model
# note that the budgets don't have to align, but beware: if the max budget of the second run is not
# greater or equal to the max budget in the previous runs, BOHB's model might never be updated!
min_budget *= 3
max_budget *= 3

HB = opt(	configspace = config_space,
				run_id = run_id,
                eta=3,min_budget=min_budget, max_budget=max_budget,
				nameserver=ns_host,
				nameserver_port = ns_port,
				previous_result = res,			# Here is now we feed the previous run into our optimizer
				min_points_in_model=4,
				top_n_percent=5,
				bandwidth_factor=1,
				num_samples=32,
				min_bandwidth=1e-1
				)
res = HB.run(4)
HB.shutdown(shutdown_workers=True)
NS.shutdown()



incumbent_trajectory = res.get_incumbent_trajectory()




def extract_HB_learning_curves(runs):
	sr = sorted(runs, key=lambda r: r.budget)
	return([[(r.time_stamps['finished'], r.loss) for r in sr],])
		

lcs = res.get_learning_curves(lc_extractor=extract_HB_learning_curves)
from hpbandster.visualization import interactive_HB_plot, default_tool_tips



tool_tips = default_tool_tips(res, lcs)
fig, ax, check, none_button, all_button = interactive_HB_plot(lcs, tool_tip_strings=tool_tips, show=False)
ax.set_ylim([0.1*incumbent_trajectory['losses'][-1], 1])
ax.set_yscale('log')

id2config = res.get_id2config_mapping()

print('A total of %i unique configurations where sampled.'%len(id2config.keys()))
print('A total of %i runs where executed.'%len(res.get_all_runs()))



plt.figure()

plt.plot(incumbent_trajectory['times_finished'], incumbent_trajectory['losses'])
plt.xlabel('wall clock time [s]')
plt.ylabel('incumbent loss')
plt.show()

