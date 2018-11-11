import os
import time
import unittest
import tempfile

import logging

logging.basicConfig(level=logging.WARNING)

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpn

from hpbandster.optimizers.hyperband import HyperBand
from hpbandster.optimizers.bohb import BOHB
from hpbandster.optimizers.h2bo import H2BO
# from hpbandster.optimizers.lcnet import LCNet
from hpbandster.optimizers.randomsearch import RandomSearch

rapid_development = True
rapid_development = False


class TestWorker(Worker):

    def __init__(self, sleep_duration=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep_duration = sleep_duration

    def compute(self, *args, **kwargs):
        time.sleep(self.sleep_duration)
        return ({'loss': 0, 'info': {}})


class TestWorkers(unittest.TestCase):
    def setUp(self):
        self.configspace = CS.ConfigurationSpace(42)
        self.configspace.add_hyperparameters([CS.UniformFloatHyperparameter('cont1', lower=0, upper=1)])

        self.run_id = 'hpbandsterUnittestWorker'

    def tearDown(self):
        self.configspace = None

    @unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
    def test_NoNameserverForeground(self):
        w = TestWorker(run_id='test')
        self.assertRaises(RuntimeError, w.run, background=False)

    @unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
    def test_NoNameserverBackground(self):
        w = TestWorker(run_id='test')
        w.run(background=True)
        w.thread.join()
        self.assertFalse(w.thread.is_alive())

    @unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
    def test_NoNameserverCredentials(self):
        w = TestWorker(run_id='test')

        with tempfile.TemporaryDirectory() as working_directory:
            self.assertRaises(RuntimeError, w.load_nameserver_credentials, working_directory, num_tries=1)

    @unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
    def test_Timeout(self):
        class dummy_callback(object):
            def register_result(self, *args, **kwargs):
                pass

        host = hpn.nic_name_to_host('lo')

        w = TestWorker(run_id=self.run_id, sleep_duration=0, timeout=1, host=host)

        dc = dummy_callback()

        with tempfile.TemporaryDirectory() as working_directory:
            # start up nameserver
            ns = hpn.NameServer(self.run_id, working_directory=working_directory, host=host)
            ns_host, ns_port = ns.start()

            # connect worker to it
            w.load_nameserver_credentials(working_directory)
            w.run(background=True)

            # start a computation with a dummy callback and dummy id
            w.start_computation(dc, '0')

            # at this point the worker must still be alive
            self.assertTrue(w.thread.is_alive())

            # as the timeout is only 1, after 2 seconds, the worker thread should be dead
            time.sleep(2)
            self.assertFalse(w.thread.is_alive())

            # shutdown the nameserver before the temporary directory is gone
            ns.shutdown()

    @unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
    def test_Timeout(self):
        host = hpn.nic_name_to_host('lo')

        with tempfile.TemporaryDirectory() as working_directory:
            # start up nameserver
            ns = hpn.NameServer(self.run_id, working_directory=working_directory, host=host)
            ns_host, ns_port = ns.start()

            # create workers and connect them to the nameserver
            workers = []
            for i in range(3):
                w = TestWorker(run_id=self.run_id, sleep_duration=2, timeout=1, host=host, id=i)
                w.load_nameserver_credentials(working_directory)
                w.run(background=True)
                workers.append(w)

            # at this point all workers must still be alive
            alive = [w.thread.is_alive() for w in workers]
            self.assertTrue(all(alive))

            opt = HyperBand(run_id=self.run_id,
                            configspace=self.configspace,
                            nameserver=ns_host,
                            nameserver_port=ns_port,
                            min_budget=1, max_budget=3, eta=3, ping_interval=1)
            opt.run(1, min_n_workers=3)

            # only one worker should be alive when the run is done
            alive = [w.thread.is_alive() for w in workers]
            self.assertEqual(1, sum(alive))

            opt.shutdown()
            time.sleep(2)

            # at this point all workers should have finished
            alive = [w.thread.is_alive() for w in workers]
            self.assertFalse(any(alive))

            # shutdown the nameserver before the temporary directory is gone
            ns.shutdown()

    @unittest.skipIf(rapid_development, "test skipped to accelerate developing new tests")
    def test_optimizers(self):
        optimizers = [BOHB, H2BO, RandomSearch]

        for optimizer in optimizers:
            host = hpn.nic_name_to_host('lo')

            with tempfile.TemporaryDirectory() as working_directory:
                # start up nameserver
                ns = hpn.NameServer(self.run_id, working_directory=working_directory, host=host)
                ns_host, ns_port = ns.start()

                # create workers and connect them to the nameserver
                w = TestWorker(run_id=self.run_id, sleep_duration=2, timeout=1, host=host, id=1)
                w.load_nameserver_credentials(working_directory)
                w.run(background=True)

                opt = optimizer(run_id=self.run_id,
                                configspace=self.configspace,
                                nameserver=ns_host,
                                nameserver_port=ns_port,
                                min_budget=1, max_budget=3, eta=3, ping_interval=1)
                opt.run(1, min_n_workers=1)

                opt.shutdown()
                time.sleep(2)

                # shutdown the nameserver before the temporary directory is gone
                ns.shutdown()


if __name__ == '__main__':
    unittest.main()
