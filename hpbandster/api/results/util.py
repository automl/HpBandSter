import os.path
import json


import hpbandster

class json_result_logger(object):
	"""
		convenience logger for 'semi-live-results'

		Logger that writes job results into two files (configs.json and results.json).
		Both files contain propper json objects in each line.

		This version (v1) opens and closes the files for each result.
		This might be very slow if individual runs are fast and the
		filesystem is rather slow (e.g. a NFS).

	"""
	def __init__(self, directory, overwrite=False):
		"""
			Parameters:
			-----------

			directory: string
				the directory where the two files 'configs.json' and
				'results.json' are stored
			overwrite: bool
				In case the files already exist, this flag controls the
				behavior:
					> True:   The existing files will be overwritten.
					          Potential risk of deleting previous results
					> False:  A FileEvistsError is raised and the files are
							  not modified.
		"""

		os.makedirs(directory, exist_ok=True)

		
		self.config_fn  = os.path.join(directory, 'configs.json')
		self.results_fn = os.path.join(directory, 'results.json')


		try:
			with open(self.config_fn, 'x') as fh: pass
		except FileExistsError:
			if overwrite:
				with open(self.config_fn, 'w') as fh: pass
			else:
				raise FileExistsError('The file %s already exists.'%self.config_fn)
		except:
			raise

		try:
			with open(self.results_fn, 'x') as fh: pass
		except FileExistsError:
			if overwrite:
				with open(self.results_fn, 'w') as fh: pass
			else:
				raise FileExistsError('The file %s already exists.'%self.config_fn)

		except:
			raise

		self.config_ids = set()

	def __call__(self, job):
		if not job.id in self.config_ids:
			self.config_ids.add(job.id)
			with open(self.config_fn, 'a') as fh:
				fh.write(json.dumps([job.id, job.kwargs['config']]))
				fh.write('\n')
		with open(self.results_fn, 'a') as fh:
			fh.write(json.dumps([job.id, job.kwargs['budget'], job.timestamps, job.result, job.exception]))
			fh.write("\n")



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

			data[tuple(config_id)] = {
									'config'     : config,
									'results'    : {},
									'time_stamps': {},
									'exceptions' : {}
								}


	with open(os.path.join(directory, 'results.json')) as fh:
		for line in fh:
			config_id, budget,time_stamps, result, exception = json.loads(line)

			id = tuple(config_id)
			data[id]['time_stamps'][budget] = time_stamps
			data[id]['results'][budget] = result
			data[id]['exceptions'][budget] = exception

			budget_set.add(budget)
			time_ref = min(time_ref, time_stamps['submitted'])


	# infere the hyperband configuration from the data
	budget_list = sorted(list(budget_set))
	
	HB_config = {
						'eta'        : None if len(budget_list) < 2 else budget_list[1]/budget_list[0],
						'min_budget' : min(budget_set),
						'max_budget' : max(budget_set),
						'budgets'    : budget_list,
						'max_SH_iter': len(budget_set),
						'time_ref'   : time_ref
				}
	return(hpbandster.HB_result([data], HB_config))



