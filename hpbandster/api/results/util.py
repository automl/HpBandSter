import os.path
import json

from hpbandster.api.results.result import Result
from hpbandster.iterations.base import Datum


def logged_results_to_HB_result(directory):
	"""
		function to import logged 'live-results' and return a HB_result object

		You can load live run results with this function and the returned
		HB_result object gives you access to the results the same way
		a finished run would.
	"""
	data = {}
	time_ref = float('inf')
	budget_set = set()
	
	with open(os.path.join(directory, 'configs.json')) as fh:
		for line in fh:
			config_id, config = json.loads(line)

			data[tuple(config_id)] = Datum(config=config, config_info='NA')

	with open(os.path.join(directory, 'results.json')) as fh:
		for line in fh:
			config_id, budget,time_stamps, result, exception = json.loads(line)

			id = tuple(config_id)
			data[id].time_stamps[budget] = time_stamps
			data[id].results[budget] = result
			data[id].exceptions[budget] = exception

			budget_set.add(budget)
			time_ref = min(time_ref, time_stamps['submitted'])


	# infer the hyperband configuration from the data
	budget_list = sorted(list(budget_set))
	
	HB_config = {
						'eta'        : None if len(budget_list) < 2 else budget_list[1]/budget_list[0],
						'min_budget' : min(budget_set),
						'max_budget' : max(budget_set),
						'budgets'    : budget_list,
						'max_SH_iter': len(budget_set),
						'time_ref'   : time_ref
				}
	return(Result([data], HB_config))



