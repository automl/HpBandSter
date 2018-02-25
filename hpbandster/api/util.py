import os
import pickle
import json
import threading

import Pyro4.naming


def nic_name_to_host(nic_name):
	""" translates the name of a network card into a valid host name"""
	from netifaces import ifaddresses, AF_INET
	host = ifaddresses(nic_name).setdefault(AF_INET, [{'addr': 'No IP addr'}] )[0]['addr']
	return(host)



class NameServer(object):

	def __init__(self, run_id, working_directory=None, host=None, port=0, nic_name=None):
		"""
		Convenience wrapper for a PYRO4 nameserver

		Parameters:
		-----------
			run_id: str
				unique run_id associated with the HPB run
			working_directory: str
				path to the working directory of the HPB run to store the nameservers credentials.
				If None, no config file will be written.
			host: str
				the hostname to use for the nameserver
			port: int
				the port to be used. Default (=0) means a random port
			nic_name: str
				name of the network interface to use (only used if host is not given)
			

		"""
		self.run_id = run_id
		self.host = host
		self.nic_name = nic_name
		self.port = port
		self.dir = working_directory
		self.conf_fn = None
		self.pyro_ns = None



	def start(self):
		"""	
		starts a Pyro4 nameserver in a separate thread
		
		Returns:
		--------
			tuple (str, int):
				the host name and the used port
		"""
	
		if self.host is None:
			if self.nic_name is None:
				self.host = 'localhost'
			else:
				self.host = nic_name_to_host(self.nic_name)

		uri, self.pyro_ns, _ = Pyro4.naming.startNS(host=self.host, port=self.port)

		self.host, self.port = self.pyro_ns.locationStr.split(':')
		self.port = int(self.port)
		
		thread = threading.Thread(target=self.pyro_ns.requestLoop, name='Pyro4 nameserver started by HpBandSter')
		#thread.daemon=True
		
		thread.start()

		if not self.dir is None:
			self.conf_fn = os.path.join(self.dir, 'HPB_run_%s_pyro.pkl'%self.run_id)

			with open(self.conf_fn, 'wb') as fh:
				pickle.dump((self.host, self.port), fh)
		
		return(self.host, self.port)


	def shutdown(self):
		"""
			clean shutdown of the nameserver and the config file (if written)
		"""
		if not self.pyro_ns is None:
			self.pyro_ns.shutdown()
			self.pyro_ns = None
		
		if not self.conf_fn is None:
			os.remove(self.conf_fn)
			self.conf_fn = None


	def __del__(self):
		self.shutdown()



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
