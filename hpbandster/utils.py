import os.path
import json
import threading

import Pyro4
import Pyro4.naming


from hpbandster.core.result import Result
from hpbandster.core.base_iteration import Datum



def nic_name_to_host(nic_name=None):
    """ translates the name of a network card into a valid host name"""
    from netifaces import ifaddresses, AF_INET

    def get_nic_name_from_system():
        import re
        import subprocess
        process = subprocess.Popen("ip route get 8.8.8.8".split(),
                                   stdout=subprocess.PIPE)
        output = process.stdout.read().decode()
        s = re.search(r'dev\s*(\S+)', output)
        return s.group(1)

    # if the network card name is not a valid one an ecxeption will be raised
    # and the method get_nic_name_from_system will discover a valid card name
    try:
        host = ifaddresses(nic_name).setdefault(AF_INET, [{'addr': 'No IP addr'}] )[0]['addr']
    # ValueError if the nic_name is no correct
    # TypeError is nic_name is None
    except (ValueError, TypeError) as e:
        nic_name = get_nic_name_from_system()
        host = ifaddresses(nic_name).setdefault(AF_INET, [{'addr': 'No IP addr'}] )[0]['addr']

    return(host)



def start_local_nameserver(host=None, port=0, nic_name=None):
	"""
		starts a Pyro4 nameserver in a daemon thread
		
		Parameters:
		-----------
			host: str
				the hostname to use for the nameserver
			port: int
				the port to be used. Default =0 means a random port
			nic_name: str
				name of the network interface to use
		
		Returns:
		--------
			tuple (str, int):
				the host name and the used port
	"""
	
	if host is None:
		if nic_name is None:
			host = 'localhost'
		else:
			host = nic_name_to_host(nic_name)

	uri, ns, _ = Pyro4.naming.startNS(host=host, port=port)
	host, port = ns.locationStr.split(':')
	
	
	thread = threading.Thread(target=ns.requestLoop, name='Pyro4 nameserver started by HpBandSter')
	thread.daemon=True
	
	thread.start()
	return(host, int(port))



