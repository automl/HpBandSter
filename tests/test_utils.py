import unittest
import logging
from io import StringIO
import sys

logging.basicConfig(level=logging.WARNING)

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpn
import hpbandster.utils as utils

rapid_development = True
rapid_development = False


class TestUtils(unittest.TestCase):

    def test_local_nameserver_1(self):
        host, port = utils.start_local_nameserver(host=None, nic_name=None)
        self.assertEqual(host, 'localhost')

        ns = hpn.NameServer('0', host=host)
        ns_host, ns_port = ns.start()
        self.assertEqual(ns.host, 'localhost')
        ns.shutdown()

    def test_local_nameserver_2(self):
        host, port = utils.start_local_nameserver(host=None, nic_name='lo')
        self.assertEqual(host, '127.0.0.1')

        ns = hpn.NameServer('0', host=host)
        ns_host, ns_port = ns.start()
        self.assertEqual(ns.host, '127.0.0.1')
        ns.shutdown()

    def test_predict_bobh_run(self):
        stdout = StringIO()
        sys.stdout = stdout
        utils.predict_bobh_run(1, 9, eta=3, n_iterations=5)
        expected = """Running BOBH with these parameters will proceed as follows:
  5 iterations of SuccessiveHalving will be executed.
  The iterations will start with a number of configurations as [9, 3, 3, 9, 3].
  With the initial budgets as [1.0, 3.0, 9, 1.0, 3.0].
  A total of 27 unique configurations will be sampled.
  A total of 37 runs will be executed.
"""
        self.assertEqual(stdout.getvalue(), expected)


if __name__ == '__main__':
    unittest.main()
