import unittest
from hpbandster.core.result import Run, extract_HBS_learning_curves, \
    json_result_logger, logged_results_to_HBS_result
from hpbandster.core.base_iteration import Datum

import ConfigSpace as CS

import tempfile
import sys
import os.path


class TestResult(unittest.TestCase):

    def test_init(self):
        run_obj = Run(config_id=1, budget=2, loss=[3, 1], info={'loss': [3, 1]},
                      time_stamps={'submitted': 0, 'started': 10}, error_logs=None)

        self.assertEqual(run_obj.config_id, 1)
        self.assertEqual(run_obj.budget, 2)
        self.assertListEqual(run_obj.loss, [3, 1])
        self.assertListEqual(run_obj.info['loss'], [3, 1])
        self.assertDictEqual(run_obj.time_stamps, {'submitted': 0, 'started': 10})

class TestExtraction(unittest.TestCase):
    def test_extract_HBS_learning_curves(self):
        run_1 = Run('1', 10, 1, {}, {}, None)
        run_2 = Run('2', 6, 3, {}, {}, None)
        # the function should filter out invalid runs --> runs with no loss value
        run_3 = Run('3', 3, None, {}, {}, None)
        run_4 = Run('4', 1, 7, {}, {}, None)

        self.assertListEqual(extract_HBS_learning_curves([run_1, run_2, run_3, run_4]),
                             [[(1, 7), (6, 3), (10, 1)]])

class TestJsonResultLogger(unittest.TestCase):
    def test_write_new_config(self):

        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CS.CategoricalHyperparameter('test', [1]))

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = json_result_logger(temp_dir)

            logger.new_config('1', cs.sample_configuration().get_dictionary(), {'test': 'test'})

            self.assertTrue(os.path.exists(temp_dir))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'configs.json')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'results.json')))
            self.assertEqual(logger.config_ids, set('1'))

            with open(os.path.join(temp_dir, 'configs.json')) as fh:
                data = fh.read()
                data = data.rstrip()
                self.assertEqual(data, r'["1", {"test": 1}, {"test": "test"}]')

"""
class TestResultObject(unittest.TestCase):
    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()
        with open(os.path.join(self.temp_dir.name, 'configs.json'), 'w') as f:
            f.write('[[0, 0, 0], {"act_f": "Tanh"}, {"model_based_pick": false}]\n')
            f.write('[[0, 0, 1], {"act_f": "ReLU"}, {"model_based_pick": false}]')

        with open(os.path.join(self.temp_dir.name, 'results.json'), 'w') as f:
            f.write('[[0, 0, 0], 5, {"submitted": 15, "started": 16, "finished": 17},'
                    ' {"loss": 7, "info": {"loss": 7}}, null]\n')
            f.write('[[0, 0, 1], 10, {"submitted": 17, "started": 18, "finished": 19},'
                    ' {"loss": 9, "info": {"loss": 9}}, null]')

    def tearDown(self):
        os.remove(os.path.join(self.temp_dir.name, 'configs.json'))
        os.remove(os.path.join(self.temp_dir.name, 'results.json'))
        #os.rmdir(self.temp_dir.name)

    def test_logged_results_to_HBS_result(self):
        # result, config = logged_results_to_HBS_result(self.temp_dir.name)
        # print(result, config)
"""
