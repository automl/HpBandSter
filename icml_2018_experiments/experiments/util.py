import os
import pickle

import numpy as np

import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import RandomSearch, BOHB, HyperBand



def standard_parser_args(parser):
	parser.add_argument('--dest_dir', type=str, help='the destination directory. A new subfolder is created for each benchmark/dataset.', default='../results/')
	parser.add_argument('--num_iterations', type=int, help='number of Hyperband iterations performed.', default=4)
	parser.add_argument('--run_id', type=str, default=0)
	parser.add_argument('--working_directory', type=str, help='Directory holding live rundata. Should be shared across all nodes for parallel optimization.', default='/tmp/')
	parser.add_argument('--method', type=str, default='randomsearch', help='Possible choices: randomsearch, bohb, hyperband, tpe, smac')
	parser.add_argument('--nic_name', type=str, default='lo', help='name of the network interface used for communication. Note: default is only for local execution on *nix!')
	parser.add_argument('--min_budget', type=float, help='Minimum budget for Hyperband and BOHB.')
	parser.add_argument('--max_budget', type=float, help='Maximum budget for all methods.')
	parser.add_argument('--eta', type=float, help='Eta value for Hyperband/BOHB.', default=3)

	return(parser)




def get_optimizer(parsed_args, config_space, **kwargs):
	
	eta = parsed_args.eta
	opt = None
	
	
	if parsed_args.method == 'randomsearch':
		opt = RandomSearch
		
	if parsed_args.method == 'bohb':
		opt = BOHB
		
	if parsed_args.method == 'hyperband':
		opt = HyperBand
	
	if opt is None:
		raise ValueError("Unknown method %s"%parsed_args.method)
	
	return(opt(config_space, eta=eta, **kwargs))



def extract_results_to_pickle(results_object):
	"""
		Returns the best configurations over time, but also returns the cummulative budget
		
		
		Parameters:
		-----------
			all_budgets: bool
				If set to true all runs (even those not with the largest budget) can be the incumbent.
				Otherwise, only full budget runs are considered
		
		Returns:
		--------
			dict:
				dictionary with all the config IDs, the times the runs
				finished, their respective budgets, and corresponding losses
	"""
	all_runs = results_object.get_all_runs(only_largest_budget = False)
	all_runs.sort(key=lambda r: r.time_stamps['finished'])
	
	return_dict = { 'config_ids' : [],
					'times_finished': [],
					'budgets'    : [],
					'losses'     : [],
					'test_losses': [],
					'cummulative_budget' : [],
					'cummulative_cost' : []
	}

	cummulative_budget = 0
	cummulative_cost = 0
	current_incumbent = float('inf')
	incumbent_budget = -float('inf')
	
	for r in all_runs:
		
		cummulative_budget += r.budget
		try: cummulative_cost += r.info['cost']
		except: pass
		
		if r.loss is None: continue
		
		if (r.budget >= incumbent_budget and r.loss < current_incumbent):
			current_incumbent = r.loss
			incumbent_budget  = r.budget
			
			return_dict['config_ids'].append(r.config_id)
			return_dict['times_finished'].append(r.time_stamps['finished'])
			return_dict['budgets'].append(r.budget)
			return_dict['losses'].append(r.loss)
			return_dict['cummulative_budget'].append(cummulative_budget)
			return_dict['cummulative_cost'].append(cummulative_cost)
			try: return_dict['test_losses'].append(r.info['test_loss'])
			except: pass


	if current_incumbent != r.loss:
		r = all_runs[-1]
	
		return_dict['config_ids'].append(return_dict['config_ids'][-1])
		return_dict['times_finished'].append(r.time_stamps['finished'])
		return_dict['budgets'].append(return_dict['budgets'][-1])
		return_dict['losses'].append(return_dict['losses'][-1])
		return_dict['cummulative_budget'].append(cummulative_budget)
		return_dict['cummulative_cost'].append(cummulative_cost)
		try: return_dict['test_losses'].append(return_dict['test_losses'][-1])
		except: pass

	return_dict['configs'] = {}
	
	id2conf = results_object.get_id2config_mapping()
	
	
	for c in return_dict['config_ids']:
		return_dict['configs'][c] = id2conf[c]
	
	return_dict['HB_config'] = results_object.HB_config
	
	return (return_dict)




def run_experiment(args, worker, dest_dir, smac_deterministic, store_all_runs=False):

	# make sure the working and dest directory exist
	os.makedirs(args.working_directory, exist_ok=True)
	os.makedirs(dest_dir, exist_ok=True)

	if args.method in ['randomsearch', 'bohb', 'hyperband']:
		# setup a nameserver
		NS = hpns.NameServer(run_id=args.run_id, nic_name=args.nic_name, working_directory=args.working_directory)
		ns_host, ns_port = NS.start()

		# start worker in the background
		worker.load_nameserver_credentials(args.working_directory)
		worker.run(background=True)

		configspace = worker.configspace


		HPB =  get_optimizer(args, configspace, working_directory=args.dest_dir,
					run_id = args.run_id,
					min_budget=args.min_budget, max_budget=args.max_budget,
					host=ns_host,
					nameserver=ns_host,
					nameserver_port = ns_port,
					ping_interval=3600,
					result_logger=None,
				)

		result = HPB.run(n_iterations = args.num_iterations) 

		# shutdown the worker and the dispatcher
		HPB.shutdown(shutdown_workers=True)
		NS.shutdown()

	# the number of iterations for the blackbox optimizers must be increased so they have comparable total budgets
	bb_iterations = int(args.num_iterations * (1+(np.log(args.max_budget) - np.log(args.min_budget))/np.log(args.eta)))

	if args.method == 'tpe':
		result = worker.run_tpe(bb_iterations)
		
	if args.method == 'smac':
		result = worker.run_smac(bb_iterations, deterministic=smac_deterministic)	

	if result is None:
		raise ValueError("Unknown method %s!"%args.method)

	with open(os.path.join(dest_dir, '{}_run_{}.pkl'.format(args.method, args.run_id)), 'wb') as fh:
		pickle.dump(extract_results_to_pickle(result), fh)
	
	if store_all_runs:
		with open(os.path.join(dest_dir, '{}_full_run_{}.pkl'.format(args.method, args.run_id)), 'wb') as fh:
			pickle.dump(extract_results_to_pickle(result), fh)
	
	# in case one wants to inspect the complete run
	return(result)
