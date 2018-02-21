import time
import logging
import pickle
import os, socket


import traceback
import threading
import Pyro4



class Worker(object):
	def __init__(self, run_id, nameserver=None, nameserver_port=None, logger=None, host=None, id=None):
		self.run_id = run_id
		self.host = host
		self.nameserver = nameserver
		self.nameserver_port = nameserver_port
		self.worker_id =  "hpbandster.run_%s.worker.%s.%i"%(self.run_id, socket.gethostname(), os.getpid())
		
		if not id is None:
			self.worker_id +='.%s'%str(id)

		self.thread=None

		if logger is None:
			logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s',  datefmt='%H:%M:%S')
			self.logger = logging.getLogger(self.worker_id)
		else:
			self.logger = logger


		
		self.busy = False
		self.thread_cond = threading.Condition(threading.Lock())


	def load_nameserver_credentials(self, working_directory, num_tries=60, interval=1):
		"""
			loads the nameserver credentials in cases where master and workers share a filesystem

			Parameters:
			-----------
				working_directory: str
					the working directory for the HPB run (see master)
				num_tries: int
					number of attempts to find the file (default 60)
				interval: float
					waiting period between the attempts
				

		"""
		fn = os.path.join(working_directory, 'HPB_run_%s_pyro.pkl'%self.run_id)
		
		for i in range(num_tries):
			try:
				with open(fn, 'rb') as fh:
					self.nameserver, self.nameserver_port = pickle.load(fh)
				return
			except FileNotFoundError:
				print('config file %s not found (trail %i/%i)'%(fn, i+1, num_tries))
				time.sleep(interval)
			except:
				raise
		raise RuntimeError("Could not find the nameserver information, aborting!")


	def run(self, background=False):
		"""
			Method to start the worker.
			
			Parameters:
			-----------
				background: bool
					If set to False (Default). the worker is executed in the current thread.
					If True, a new daemon thread is created that runs the worker. This is
					useful in a single worker scenario/when the compute function only simulates
					work.
		"""
		if background:
			self.worker_id += str(threading.get_ident())
			self.thread = threading.Thread(target=self._run, name='worker %s thread'%self.worker_id)
			self.thread.daemon=True
			self.thread.start()
		else:
			self._run()

	def _run(self):
		# initial ping to the dispatcher to register the worker
		with Pyro4.locateNS(host=self.nameserver, port=self.nameserver_port) as ns:
			self.logger.debug('WORKER: Connected to nameserver %s'%(str(ns)))
			dispatchers = ns.list(prefix="hpbandster.run_%s.dispatcher"%self.run_id)

		for dn, uri in dispatchers.items():
			try:
				self.logger.debug('WORKER: found dispatcher %s'%dn)
				with Pyro4.Proxy(uri) as dispatcher_proxy:
					dispatcher_proxy.trigger_discover_worker()

			except Pyro4.errors.CommunicationError:
				self.logger.debug('WORKER: Dispatcher did not respond. Waiting for one to initiate contact.')
				pass
			except:
				raise

		if len(dispatchers) == 0:
			self.logger.debug('WORKER: No dispatcher found. Waiting for one to initiate contact.')

		self.logger.info('WORKER: start listening for jobs')

		self.pyro_daemon = Pyro4.core.Daemon(host=self.host)

		with Pyro4.locateNS(self.nameserver, port=self.nameserver_port) as ns:
			uri = self.pyro_daemon.register(self, self.worker_id)
			ns.register(self.worker_id, uri)
		
		self.pyro_daemon.requestLoop()

		with Pyro4.locateNS(self.nameserver, port=self.nameserver_port) as ns:
			ns.remove(self.worker_id)
		
		

	def compute(self, *args, **kwargs):
		raise NotImplementedError("Subclass hpbandster.distributed.worker and overwrite the compute method in your worker script")

	@Pyro4.expose
	@Pyro4.oneway
	def start_computation(self, callback, id, *args, **kwargs):

		with self.thread_cond:
			while self.busy:
				self.thread_cond.wait()
			self.busy = True
		self.logger.info('WORKER: start processing job %s'%str(id))
		self.logger.debug('WORKER: args: %s'%(str(args)))
		self.logger.debug('WORKER: kwargs: %s'%(str(kwargs)))
		try:
			result = {'result': self.compute(*args, **kwargs),
						'exception' : None}
		except Exception as e:
			result = {'result': None,
						'exception' : traceback.format_exc()}
		finally:
			self.logger.debug('WORKER: done with job %s, trying to register it.'%str(id))
			with self.thread_cond:
				self.busy =  False
				callback.register_result(id, result)
				self.thread_cond.notify()
		self.logger.info('WORKER: registered result for job %s with dispatcher'%str(id))
		return(result)

	@Pyro4.expose	
	def is_busy(self):
		return(self.busy)
	
	@Pyro4.expose
	@Pyro4.oneway
	def shutdown(self):
		self.pyro_daemon.shutdown()
		if not self.thread is None:
			self.thread.join()
