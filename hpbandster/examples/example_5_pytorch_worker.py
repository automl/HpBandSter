"""
Worker for Example 5 - PyTorch
==============================

In this example implements a small CNN in PyTorch to train it on MNIST.
The configuration space shows the most common types of hyperparameters and
even contains conditional dependencies.
In this example implements a small CNN in Keras to train it on MNIST.
The configuration space shows the most common types of hyperparameters and
even contains conditional dependencies.

We'll optimise the following hyperparameters:

+-------------------------+----------------+-----------------+------------------------+
| Parameter Name          | Parameter type |  Range/Choices  | Comment                |
+=========================+================+=================+========================+
| Learning rate           |  float         | [1e-6, 1e-2]    | varied logarithmically |
+-------------------------+----------------+-----------------+------------------------+
| Optimizer               | categorical    | {Adam, SGD }    | discrete choice        |
+-------------------------+----------------+-----------------+------------------------+
| SGD momentum            |  float         | [0, 0.99]       | only active if         |
|                         |                |                 | optimizer == SGD       |
+-------------------------+----------------+-----------------+------------------------+
| Number of conv layers   | integer        | [1,3]           | can only take integer  |
|                         |                |                 | values 1, 2, or 3      |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | logarithmically varied |
| the first conf layer    |                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the second conf layer   |                |                 | of layers >= 2         |
+-------------------------+----------------+-----------------+------------------------+
| Number of filters in    | integer        | [4, 64]         | only active if number  |
| the third conf layer    |                |                 | of layers == 3         |
+-------------------------+----------------+-----------------+------------------------+
| Dropout rate            |  float         | [0, 0.9]        | standard continuous    |
|                         |                |                 | parameter              |
+-------------------------+----------------+-----------------+------------------------+
| Number of hidden units  | integer        | [8,256]         | logarithmically varied |
| in fully connected layer|                |                 | integer values         |
+-------------------------+----------------+-----------------+------------------------+

Please refer to the compute method below to see how those are defined using the
ConfigSpace package.
	  
The network does not achieve stellar performance when a random configuration is samples,
but a few iterations should yield an accuracy of >90%. To speed up training, only
8192 images are used for training, 1024 for validation.
The purpose is not to achieve state of the art on MNIST, but to show how to use
PyTorch inside HpBandSter, and to demonstrate a more complicated search space.
"""

try:
	import torch
	import torch.utils.data
	import torch.nn as nn
	import torch.nn.functional as F
except:
	raise ImportError("For this example you need to install pytorch.")

try:
	import torchvision
	import torchvision.transforms as transforms
except:
	raise ImportError("For this example you need to install pytorch-vision.")



import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)



class PyTorchWorker(Worker):
	def __init__(self, N_train = 8192, N_valid = 1024, **kwargs):
		super().__init__(**kwargs)

		batch_size = 64

		# Load the MNIST Data here
		train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(), download=True)
		test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())
		
		train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(N_train))
		validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(N_train, N_train+N_valid))

		
		self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)
		self.validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024, sampler=validation_sampler)

		self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)


	def compute(self, config, budget, working_directory, *args, **kwargs):
		"""
		Simple example for a compute function using a feed forward network.
		It is trained on the MNIST dataset.
		The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
		"""

		# device = torch.device('cpu')
		model = MNISTConvNet(num_conv_layers=config['num_conv_layers'],
							num_filters_1=config['num_filters_1'],
							num_filters_2=config['num_filters_2'] if 'num_filters_2' in config else None,
							num_filters_3=config['num_filters_3'] if 'num_filters_3' in config else None,
							dropout_rate=config['dropout_rate'],
							num_fc_units=config['num_fc_units'],
							kernel_size=3
		)

		criterion = torch.nn.CrossEntropyLoss()
		if config['optimizer'] == 'Adam':
			optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
		else:
			optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

		for epoch in range(int(budget)):
			loss = 0
			model.train()
			for i, (x, y) in enumerate(self.train_loader):
				optimizer.zero_grad()
				output = model(x)
				loss = F.nll_loss(output, y)
				loss.backward()
				optimizer.step()

		train_accuracy = self.evaluate_accuracy(model, self.train_loader)
		validation_accuracy = self.evaluate_accuracy(model, self.validation_loader)
		test_accuracy = self.evaluate_accuracy(model, self.test_loader)

		return ({
			'loss': 1-validation_accuracy, # remember: HpBandSter always minimizes!
			'info': {	'test accuracy': test_accuracy,
						'train accuracy': train_accuracy,
						'validation accuracy': validation_accuracy,
						'number of parameters': model.number_of_parameters(),
					}
						
		})

	def evaluate_accuracy(self, model, data_loader):
		model.eval()
		correct=0
		with torch.no_grad():
			for x, y in data_loader:
				output = model(x)
				#test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
				pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(y.view_as(pred)).sum().item()
		#import pdb; pdb.set_trace()	
		accuracy = correct/len(data_loader.sampler)
		return(accuracy)


	@staticmethod
	def get_configspace():
		"""
		It builds the configuration space with the needed hyperparameters.
		It is easily possible to implement different types of hyperparameters.
		Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
		:return: ConfigurationsSpace-Object
		"""
		cs = CS.ConfigurationSpace()

		lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

		# For demonstration purposes, we add different optimizers as categorical hyperparameters.
		# To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
		# SGD has a different parameter 'momentum'.
		optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

		sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

		cs.add_hyperparameters([lr, optimizer, sgd_momentum])

		# The hyperparameter sgd_momentum will be used,if the configuration
		# contains 'SGD' as optimizer.
		cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
		cs.add_condition(cond)

		num_conv_layers =  CSH.UniformIntegerHyperparameter('num_conv_layers', lower=1, upper=3, default_value=2)
		
		num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=4, upper=64, default_value=16, log=True)
		num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=64, default_value=16, log=True)
		num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=64, default_value=16, log=True)


		cs.add_hyperparameters([num_conv_layers, num_filters_1, num_filters_2, num_filters_3])
		
		# You can also use inequality conditions:
		cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
		cs.add_condition(cond)

		cond = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
		cs.add_condition(cond)


		dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
		num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=256, default_value=32, log=True)

		cs.add_hyperparameters([dropout_rate, num_fc_units])

		return cs




class MNISTConvNet(torch.nn.Module):
	def __init__(self, num_conv_layers, num_filters_1, num_filters_2, num_filters_3, dropout_rate, num_fc_units, kernel_size):
		super().__init__()
		
		self.conv1 = nn.Conv2d(1, num_filters_1, kernel_size=kernel_size)
		self.conv2 = None
		self.conv3 = None
		
		output_size = (28-kernel_size + 1)//2
		num_output_filters = num_filters_1
		
		if num_conv_layers > 1:
			self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=kernel_size)
			num_output_filters = num_filters_2
			output_size = (output_size - kernel_size + 1)//2

		if num_conv_layers > 2:
			self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=kernel_size)
			num_output_filters = num_filters_3
			output_size = (output_size - kernel_size + 1)//2
		
		self.dropout = nn.Dropout(p = dropout_rate)

		self.conv_output_size = num_output_filters*output_size*output_size

		self.fc1 = nn.Linear(self.conv_output_size, num_fc_units)
		self.fc2 = nn.Linear(num_fc_units, 10)
		


	def forward(self, x):
		
		# switched order of pooling and relu compared to the original example
		# to make it identical to the keras worker
		# seems to also give better accuracies
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		
		if not self.conv2 is None:
			x = F.max_pool2d(F.relu(self.conv2(x)), 2)

		if not self.conv3 is None:
			x = F.max_pool2d(F.relu(self.conv3(x)), 2)

		x = self.dropout(x)
		
		x = x.view(-1, self.conv_output_size)
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)


	def number_of_parameters(self):
		return(sum(p.numel() for p in self.parameters() if p.requires_grad))



if __name__ == "__main__":
	worker = PyTorchWorker(run_id='0')
	cs = worker.get_configspace()
	
	config = cs.sample_configuration().get_dictionary()
	print(config)
	res = worker.compute(config=config, budget=2, working_directory='.')
	print(res)
