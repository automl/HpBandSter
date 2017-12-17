import sys
sys.path.append('..')

import unittest
import random



import hpbandster as hb



skip_most = False
skip_reason = "rapid development with only a few tests active"

class TestHB_iterations(unittest.TestCase):

	def setUp(self):
		pass
	def tearDown(self):
		pass
	
	@unittest.skipIf(skip_most, skip_reason)
	def test_SH_a_few_fails(self):
		
		
		it = hb.HB_iteration.SuccessiveHalving(0,[4,2], [1,2], lambda x: (random.random(),''))
		
		ids = []
		
		for i in range(4):
			ids.append(it.add_configuration())
			self.assertEqual(it.actual_num_configs, [i+1,0])

		confs = [it.data[i]['config'] for i in ids]


		for i in range(4):
			id,c,b = it.get_next_run()
			self.assertEqual(it.num_running, 1)

			fake_job_id = {'config_id': id, 'budget': b, 'config': c, 'time_stamps': None}
			fake_result = {'loss': i, 'info': []}
			
			if i == 3:
				fake_result = None
		
			it.register_result(fake_job_id, fake_result)
		
		it.process_results()
		
		self.assertTrue(it.SH_iter, 1)
		self.assertEqual(it.actual_num_configs, [4,2])
		
		for i in range(2):
			id,c,b = it.get_next_run()
			self.assertEqual(it.num_running, 1)

			self.assertTrue(id in [(0,0,0), (0,0,1)])
			
			fake_job_id = {'config_id': id, 'budget': b, 'config': c, 'time_stamps': None}
			fake_result = {'loss': i, 'info': []}		
		
			it.register_result(fake_job_id, fake_result)

		self.assertTrue(it.data[(0,0,3)]['results'][1] is None)


	def test_SH_many_fails(self):
		
		it = hb.HB_iteration.SuccessiveHalving(0,[4,2], [1,2], lambda x: random.random())
		
		ids = []
		
		for i in range(4):
			ids.append(it.add_configuration())
			self.assertEqual(it.actual_num_configs, [i+1,0])

		confs = [it.data[i]['config'] for i in ids]


		for i in range(4):
			id,c,b = it.get_next_run()
			self.assertEqual(it.num_running, 1)

			fake_job_id = {'config_id': id, 'budget': b, 'config': c, 'time_stamps': None}
			fake_result = {'loss': i, 'info': []}
			
			if i != 3:
				fake_result = None
		
			it.register_result(fake_job_id, fake_result)
		
		it.process_results()
		
		self.assertTrue(it.SH_iter, 1)
		self.assertEqual(it.actual_num_configs, [4,1])
		
		id,c,b = it.get_next_run()

		self.assertTrue(id == (0,0,3))
			
		fake_job_id = {'config_id': id, 'budget': b, 'config': c, 'time_stamps': None}
		fake_result = {'loss': 3, 'info': []}		
		it.register_result(fake_job_id, fake_result)

		id,c,b = it.get_next_run()
		self.assertTrue(id == (0,1,1))
			
		fake_job_id = {'config_id': id, 'budget': b, 'config': c, 'time_stamps': None}
		fake_result = {'loss': 4, 'info': []}		
		it.register_result(fake_job_id, fake_result)
		
		it.process_results()
		
		self.assertTrue(it.is_finished)



	def test_SH_all_fails(self):
		
		it = hb.HB_iteration.SuccessiveHalving(0,[4,2], [1,2], lambda x: random.random())
		
		ids = []
		
		for i in range(4):
			ids.append(it.add_configuration())
			self.assertEqual(it.actual_num_configs, [i+1,0])

		confs = [it.data[i]['config'] for i in ids]


		for i in range(4):
			id,c,b = it.get_next_run()
			self.assertEqual(it.num_running, 1)

			fake_job_id = {'config_id': id, 'budget': b, 'config': c, 'time_stamps': None}
			fake_result = None
		
			it.register_result(fake_job_id, fake_result)
		
		it.process_results()
		
		self.assertTrue(it.SH_iter, 1)
		self.assertEqual(it.actual_num_configs, [4,0])
		
		id,c,b = it.get_next_run()

		self.assertTrue(id == (0,1,0))
			
		fake_job_id = {'config_id': id, 'budget': b, 'config': c, 'time_stamps': None}
		fake_result = None
		it.register_result(fake_job_id, fake_result)

		id,c,b = it.get_next_run()
		self.assertTrue(id == (0,1,1))
			
		fake_job_id = {'config_id': id, 'budget': b, 'config': c, 'time_stamps': None}
		fake_result = None	
		it.register_result(fake_job_id, fake_result)
		
		it.process_results()
		self.assertTrue(it.is_finished)


if __name__ == "__main__":
    unittest.main()
