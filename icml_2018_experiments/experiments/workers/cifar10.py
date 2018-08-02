import time
import os, sys
import tempfile
import shutil
import subprocess
import json

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker






class CIFAR10_SSCO(Worker):

	def __init__ (self, nGPU=1, torch_source_path=None, **kwargs):
		super().__init__(**kwargs)
		self.nGPU = nGPU
		
		if torch_source_path is None:
			pwd = os.path.dirname(os.path.abspath(__file__))
			self.torch_path = os.path.join(pwd, 'lib', 'cifar10_cutout_validation')
		else:
			self.torch_path = tourch_source_path

	def compute(self, config, budget,config_id, working_directory):
		return(shakeshake_cutout_cifar10(self.complete_config(config), int(budget), working_directory, torch_source=self.torch_path, nGPU=self.nGPU))


	def get_config_space(parsed_args):
		config_space=CS.ConfigurationSpace()


		config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(		'learning_rate',lower=1e-3,	upper=1,	log=True))
		config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter(	'batch_size',	lower=32,	upper=256,	log=True))

		config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(		'weight_decay',	lower=1e-5,	upper=1e-3,	log=True))
		config_space.add_hyperparameter(CSH.UniformFloatHyperparameter(		'momentum',		lower=1e-3,	upper=0.99,	log=False))

		config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter(	'cutout_half_size',	lower=0,	upper=10,	log=False))
		return(config_space)

	def complete_config(self, config):
		""" fills in the missing entries in the config that are fixed. """
		config['depth'] =  20
		config['base_width'] = 64
		config['nCycles'] = 1
		return(config)



def load_data(dest_dir, include_learning_rate=False):
	""" helper function to read the output of the torch code """
	#training_loss, validation_loss, training_accuracy, validation_accuracy for the last epoch

	info = {}

	with open(os.path.join(dest_dir,'results.txt'), 'r') as fh:
		data = [json.loads(line) for line in fh.readlines()]
	

	with open(os.path.join(dest_dir,'log.txt'), 'r') as fh:
		 info['config'] = '\n'.join(fh.readlines())


	if include_learning_rate:
		with open(os.path.join(dest_dir,'lr.txt'), 'r') as fh:
			info['learning_rates'] = [json.loads(line)['learningRate'] for line in fh.readlines()]
	
	info['loss'] = [d['trainLoss'] for d in data]
	info['error'] = [d['trainTop1'] for d in data]
	info['val_error'] = [d['valTop1'] for d in data]
	info['test_error'] = [d['testTop1'] for d in data]
	

	return(info)



def shakeshake_cutout_cifar10(config, budget, directory, torch_source='', nGPU=1):
	""" uses Gastaldi's shakeshake code to train with the given config
	
		Note: directory is ignored and a temporary directory is created and finally removed
	"""
	
	tmp_dir = tempfile.mkdtemp(dir='/tmp/')
	dest_dir = tmp_dir

	ret_dict =  { 'loss': float('inf'), 'info': None}

	try:

		bash_strings = ["cd %s; th main.lua -dataset cifar10 -nGPU %i -nThreads %i"%(torch_source, nGPU, 2*nGPU-1),
					"-batchSize {batch_size} -depth {depth}".format(**config),
					"-baseWidth {base_width} -weightDecay {weight_decay}".format(**config),
					"-cutout_half_size {cutout_half_size}".format(**config),
					"-LR {learning_rate} -momentum {momentum}".format(**config),
					"-save {} -nEpochs {} ".format(dest_dir, budget),
					"-nCycles {nCycles}".format(**config), 
					"-netType shakeshake -lrShape cosine",
					"-shareGradInput false -optnet true",
					"-forwardShake true -backwardShake true",
					"-shakeImage true"]

		subprocess.check_call( " ".join(bash_strings), shell=True)
		info = load_data(dest_dir)

		ret_dict = { 'loss': info['val_error'][-1], 'info': info}

	except:
		raise

	finally:
		shutil.rmtree(tmp_dir, ignore_errors=True)

	return (ret_dict)


if __name__ == "__main__":
	w = CIFAR10_SSCO(run_id='bla')
	cs = w.get_config_space()
	config = cs.sample_configuration().get_dictionary()
	
	res = w.compute(config=config, budget=1, working_directory='/tmp', config_id='test')
	print(res)
