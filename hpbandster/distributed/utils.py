import threading


import Pyro4
import Pyro4.naming



def start_local_nameserver(host='localhost', port=0):
	"""
		starts a Pyro4 nameserver in a daemon thread
		
		Parameters:
		-----------
			host: str
				the hostname to use for the nameserver
			port: int
				the port to be used. Default =0 means a random port
		
		Returns:
		--------
			tuple (str, int):
				the host name and the used port
	"""
	uri, ns, _ = Pyro4.naming.startNS(host=host, port=port)
	host, port = ns.locationStr.split(':')
	
	
	thread = threading.Thread(target=ns.requestLoop, name='Pyro4 nameserver started by HpBandSter')
	thread.daemon=True
	
	thread.start()
	return(host, int(port))
