"""
Example 6: Analysis of a run
============================



"""

import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis



# load the example run from the log files
result = hpres.logged_results_to_HB_result('example_5_run/')

# get all executed runs
all_runs = result.get_all_runs()

# get the 'dict' that translates config ids to the actual configurations
id2conf = result.get_id2config_mapping()


# Here is how you get he incumbent (best configuration)
inc_id = result.get_incumbent_id()

# let's grab the run on the highest budget 
inc_runs = result.get_runs_by_id(inc_id)
inc_run = inc_runs[-1]


# We have access to all information: the config, the loss observed during
#optimization, and all the additional information
inc_loss = inc_run.loss
inc_config = id2conf[inc_id]['config']
inc_test_loss = inc_run.info['test accuracy']

print('Best found configuration:')
print(inc_config)
print('It achieved accuracies of %f (validation) and %f (test).'%(1-inc_loss, inc_test_loss))



hpvis.losses_over_time(all_runs) # plots the observed losses grouped by the budgets

hpvis.concurrent_runs_over_time(all_runs) # plots the number of concurent runs

hpvis.finished_runs_over_time(all_runs)  # see how many runs finished

hpvis.correlation_across_budgets(result) # computes the rank correlations between all the budgets

hpvis.performance_histogram_model_vs_random(all_runs, id2conf) # to compare the performance of configs picked by the model vs. random ones


plt.show()
