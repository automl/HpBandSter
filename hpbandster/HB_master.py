import os
import threading
import time
import math
import pdb
import copy
import logging

import numpy as np


from hpbandster.distributed.dispatcher import Dispatcher
from hpbandster.HB_iteration import SuccessiveHalving
from hpbandster.HB_result import HB_result

class HpBandSter(object):
	def __init__(self,
					run_id,
					config_generator,
					working_directory='.',
					eta=3, min_budget=0.01, max_budget=1,
					ping_interval=60,
					nameserver='127.0.0.1',
					ns_port=None,
					host=None,
					shutdown_workers=True,
					job_queue_sizes=(0,1),
					dynamic_queue_size=False,
					logger=None
					):
		"""

		Parameters
		----------
		run_id : string
			A unique identifier of that Hyperband run. Use the cluster's JobID when running multiple
			concurrent runs to separate them
		config_generator: hpbandster.config_generators object
			An object that can generate new configurations and registers results of executed runs
		working_directory: string
			The top level working directory accessible to all compute nodes(shared filesystem).
		eta : float
			In each iteration, a complete run of sequential halving is executed. In it,
			after evaluating each configuration on the same subset size, only a fraction of
			1/eta of them 'advances' to the next round.
			Must be greater or equal to 2.
		min_budget : float
			The smallest budget to consider. Needs to be positive!
		max_budget : float
			the largest budget to consider. Needs to be larger than min_budget!
			The budgets will be geometrically distributed $\sim \eta^k$ for
			$k\in [0, 1, ... , num_subsets - 1]$.
		ping_interval: int
			number of seconds between pings to discover new nodes. Default is 60 seconds.
		nameserver: str
			address of the Pyro4 nameserver
		ns_port: int
			port of Pyro4 nameserver
		host: str
			ip (or name that resolves to that) of the network interface to use
		shutdown_workers: bool
			flag to control whether the workers are shutdown after the computation is done
		job_queue_size: tuple of ints
			min and max size of the job queue. During the run, when the number of jobs in the queue
			reaches the min value, it will be filled up to the max size. Default: (0,1)
		dynamic_queue_size: bool
			Whether or not to change the queue size based on the number of workers available.
			If true (default), the job_queue_sizes are relative to the current number of workers.

		"""

		self.working_directory = working_directory
		os.makedirs(self.working_directory, exist_ok=True)
		

		if logger is None:
			self.logger = logging.getLogger('hpbandster')
		else:
			self.logger = logger


		self.config_generator = config_generator
		self.time_ref = None


		# Hyperband related stuff
		self.eta = eta
		self.min_budget = min_budget
		self.max_budget = max_budget


		# precompute some HB stuff
		self.max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
		self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter-1, 0, self.max_SH_iter))

		self.iterations = []
		self.jobs = []

		self.num_running_jobs = 0
		self.job_queue_sizes = job_queue_sizes
		self.user_job_queue_sizes = job_queue_sizes
		self.dynamic_queue_size = dynamic_queue_size

		if job_queue_sizes[0] >= job_queue_sizes[1]:
			raise ValueError("The queue size range needs to be (min, max) with min<max!")


		# condition to synchronize the job_callback and the queue
		self.thread_cond = threading.Condition()

		self.config = {
						'eta'        : eta,
						'min_budget' : min_budget,
						'max_budget' : max_budget,
						'budgets'    : self.budgets,
						'max_SH_iter': self.max_SH_iter,
						'time_ref'   : self.time_ref
					}


		self.dispatcher = Dispatcher( self.job_callback, queue_callback=self.adjust_queue_size, run_id=run_id, ping_interval=ping_interval, nameserver=nameserver, ns_port=ns_port, host=host)

		self.dispatcher_thread = threading.Thread(target=self.dispatcher.run)
		self.dispatcher_thread.start()


	def shutdown(self, shutdown_workers=False):
		self.logger.debug('HBMASTER: shutdown initiated, shutdown_workers = %s'%(str(shutdown_workers)))
		self.dispatcher.shutdown(shutdown_workers)
		self.dispatcher_thread.join()

	def run(self, n_iterations, iteration_class=SuccessiveHalving, min_n_workers=1, iteration_class_kwargs={}):
		"""
			method to run n_iterations of SuccessiveHalving

			Parameters:
			-----------
			n_iterations: int
				number of iterations to be performed in this run
			iteration_class: SuccessiveHalving like class
				class that runs an iteration of SuccessiveHalving or a similar
				algorithm. The API is defined by the SuccessiveHalving implementation
			min_n_workers: int
				minimum number of workers present before the run starts
			iteration_class_kwargs: dict
				Additional keyward arguments passed to iteration_class

		"""


		while (self.dispatcher.number_of_workers() < min_n_workers):
			self.logger.debug('HBMASTER: only %i worker(s) available, waiting for at least %i.'%(self.dispatcher.number_of_workers(), min_n_workers))
			time.sleep(1)

		if self.time_ref is None:
			self.time_ref = time.time()
			self.config['time_ref'] = self.time_ref
		
			self.logger.info('HBMASTER: starting run at %s'%(str(self.time_ref)))
		
		for it in range(len(self.iterations), len(self.iterations)+n_iterations):
			# number of SH iterations
			s = self.max_SH_iter - 1 - (it%self.max_SH_iter)
			# number of configurations in that bracket
			n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
			ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]

			self.iterations.append(iteration_class(iter_number=it, num_configs=ns, budgets=self.budgets[(-s-1):], config_sampler=self.config_generator.get_config, **iteration_class_kwargs))

		while len(self.active_iterations()) > 0:
			# find a new run to start
			for i in self.active_iterations():
				next_run = self.iterations[i].get_next_run()
				if not next_run is None:
					self.logger.debug('HBMASTER: schedule new run for iteration %i'%i)
					self._submit_job(*next_run)
					break # makes sure that iterations with lower numbers are scheduled first

		return HB_result([copy.deepcopy(i.data) for i in self.iterations], self.config)


	def adjust_queue_size(self, number_of_workers=None):
		if self.dynamic_queue_size:
			self.logger.debug('HBMASTER: adjusting queue size, number of workers %s'%str(number_of_workers))
			with self.thread_cond:
				nw = self.dispatcher.number_of_workers() if number_of_workers is None else number_of_workers
				self.job_queue_sizes = (self.user_job_queue_sizes[0] + nw, self.user_job_queue_sizes[1] + nw)
				self.logger.info('HBMASTER: adjusted queue size to %s'%str(self.job_queue_sizes))
				self.thread_cond.notify_all()


	def job_callback(self, job):
		"""
			method to be called when a job has finished

			this will do some book keeping and call the user defined
			new_result_callback if one was specified
		"""
		self.logger.debug('job_callback for %s'%str(job.id))
		with self.thread_cond:
			self.num_running_jobs -= 1

			if self.num_running_jobs <= self.job_queue_sizes[0]:
				self.logger.debug("HBMASTER: Trying to run another job!")
				self.thread_cond.notify()

			self.iterations[job.id[0]].register_result(job)
		self.config_generator.new_result(job)


	def _submit_job(self, config_id, config, budget):
		"""
			hidden function to submit a new job to the dispatcher

			This function handles the actual submission, keeping the number
			of jobs in the queue in the job_queue_sizes range
		"""
		self.thread_cond.acquire()
		self.logger.debug('HBMASTER: submitting job %s to dispatcher'%str(config_id))
		if self.num_running_jobs >= self.job_queue_sizes[1]:
			while(self.num_running_jobs > self.job_queue_sizes[0]):
				self.logger.debug('HBMASTER: running jobs: %i, queue sizes: %s -> wait'%(self.num_running_jobs, str(self.job_queue_sizes)))
				self.thread_cond.wait()
		self.num_running_jobs += 1
		self.thread_cond.release()

		#shouldn't the next line be executed while holding the condition?
		job = self.dispatcher.submit_job(config_id, config=config, budget=budget, working_directory=self.working_directory)
		self.logger.debug("HBMASTER: job %s submitted to dispatcher"%str(config_id))

	def active_iterations(self):
		""" function that returns a list of all iterations that are not marked as finished """
		return(list(filter(lambda idx: not self.iterations[idx].is_finished, range(len(self.iterations)))))

	def __del__(self):
		pass
