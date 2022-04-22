import unittest
import logging

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

if __name__ == '__main__':
    unittest.main()
