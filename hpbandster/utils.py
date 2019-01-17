import os.path
import json

import threading
import numpy as np
import Pyro4
import Pyro4.naming

from hpbandster.core.result import Result
from hpbandster.core.base_iteration import Datum


def nic_name_to_host(nic_name):
    """ translates the name of a network card into a valid host name"""
    from netifaces import ifaddresses, AF_INET
    host = ifaddresses(nic_name).setdefault(AF_INET, [{'addr': 'No IP addr'}])[0]['addr']
    return (host)


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
    thread.daemon = True

    thread.start()
    return (host, int(port))


def predict_bobh_run(min_budget, max_budget, eta, n_iterations):
    """
    Prints the expected numbers of configurations, runs and budgets given BOBH's hyperparameters.

    Parameters
    ----------
    min_budget
        The smallest budget to consider.
    max_budget
        The largest budget to consider.
    eta
        The eta parameter. Determines how many configurations advance to the next round.
    n_iterations
        How many iterations of SuccessiveHalving to perform.
    """
    s_max = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1

    n_runs = 0
    n_configurations = []
    initial_budgets = []
    for iteration in range(n_iterations):
        s = s_max - 1 - (iteration % s_max)

        initial_budget = (eta ** -s) * max_budget
        initial_budgets.append(initial_budget)

        n0 = int(np.floor(s_max / (s + 1)) * eta ** s)
        n_configurations.append(n0)
        ns = [max(int(n0 * (eta ** (-i))), 1) for i in range(s + 1)]
        n_runs += sum(ns)

    print('Running BOBH with these parameters will proceed as follows:')
    print('  {} iterations of SuccessiveHalving will be executed.'.format(n_iterations))
    print('  The iterations will start with a number of configurations as {}.'.format(n_configurations))
    print('  With the initial budgets as {}.'.format(initial_budgets))
    print('  A total of {} unique configurations will be sampled.'.format(sum(n_configurations)))
    print('  A total of {} runs will be executed.'.format(n_runs))
