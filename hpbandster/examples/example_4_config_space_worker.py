"""
Example 4 - How to use the configuration space
==============================================

"""
import torchvision
import torchvision.transforms as transforms

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import time
import numpy as np
import random

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)

# In this example, we'll show the example use of the ConfigSpace-Module
# It is for example used in our optimizer (e.g. SMAC or BOHB)

# A ConfigSpace object organizes the hyperparamters to be optimized.
# It offers the functionality to sample configurations from the defined configurationspace.
# It is also simple to distinguish between different hyperparameter types, like integer, float or categorical
# hyperparamters.
# A powerful advantage in comparison to naive implementations is the ability to realize conditional hyperparameters.

# In this tutorial, you'll see how to:
#   - how to connect a worker with a neural network.
#   - create a configurations space
#   - add hyperparameters of type float, integer, categorical to the configSpace
#   - and how to create conditional hyperparameters

# For demonstration purpose we'll train a one to three hidden layer neural network with either adam or sgd optimizer
# So we'll optimise the following hyperparameters:
#     - learning rate:             (float)        [1e-6, 1e-2]
#     - optimizer:                 (categorical)  ['Adam', 'SGD']
#       > sgd momentum:            (float)        [0.0, 0.99]        # only if optimizer == 'SGD'
#     - number of hidden layers    (int)          [1,3]
#       > dimension hidden layer 1 (int)          [100, 1000]
#       > dimension hidden layer 2 (int)          [100, 1000]        # only if number of hidden layers > 1
#       > dimension hidden layer 3 (int)          [100, 1000]        # only if number of hidden layers > 2
#


class MyWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        batch_size = 64

        # Load the MNIST Data here
        train_dataset = torchvision.datasets.MNIST(root='../../data',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=True)

        test_dataset = torchvision.datasets.MNIST(root='../../data',
                                                  train=False,
                                                  transform=transforms.ToTensor())

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset

        The compute-method will be called repeatedly by the BOHB optimizer. So this is the place where the
        network will be trained.
        The configuration input parameter contains the sampled hyperparameters from the configurations space

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        model = NeuralNet(input_dim=28*28,
                          num_hidden_layers=config['num_hidden_layers'],
                          hidden_dim_1=config['hidden_dim_1'],
                          hidden_dim_2=config['hidden_dim_2'] if 'hidden_dim_2' in config else None,
                          hidden_dim_3=config['hidden_dim_3'] if 'hidden_dim_3' in config else None,
                          output_dim=10,
                          act_f=config['act_f']
                          )

        criterion = torch.nn.CrossEntropyLoss()
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

        loss = 0
        for epoch in range(int(budget)):
            loss = 0

            optimizer.zero_grad()

            for i, (x, y) in enumerate(self.train_loader):
                x = x.reshape(-1, 28*28)

                output = model(x)
                loss += criterion(output, y)

            loss.backward()
            optimizer.step()

            print('Epoch [{:4}|{:4}] Loss: {:10.4f}'
                  .format(epoch+1, int(budget), loss))

        return ({
            'loss': loss.item(),  # this is the a mandatory field to run hyperband
            'info': {'loss': 'value you like to store'}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():
        """
        Here we define the configuration space for the hyperparameters for the model.
        Returns:
            ConfigSpace-object
        """
        cs = CS.ConfigurationSpace()

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
            'lr', lower=1e-6, upper=1e-2, default_value='1e-2', log=True))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(
            'act_f', ['ReLU', 'Tanh'], default_value='ReLU'))

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer',
                                                  ['Adam', 'SGD'])
        cs.add_hyperparameter(optimizer)

        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum',
                                                      lower=0.0, upper=0.99, default_value=0.9,
                                                      log=False)
        cs.add_hyperparameter(sgd_momentum)

        # The hyperparameter sgd_momentum will be used,
        # if the configuration contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)

        # The hyperparameters (hidden units for layer 2 and 3) are conditional parameters conditioned by
        # the number of hidden layers.
        # These dependencies are realised with inequality conditions.
        num_hidden_layers = CSH.UniformIntegerHyperparameter('num_hidden_layers', lower=1, upper=3, default_value=1)
        cs.add_hyperparameter(num_hidden_layers)

        hidden_dim_1 = CSH.UniformIntegerHyperparameter('hidden_dim_1', lower=100, upper=1000, log=False)
        cs.add_hyperparameter(hidden_dim_1)

        hidden_dim_2 = CSH.UniformIntegerHyperparameter('hidden_dim_2', lower=100, upper=1000, log=False)
        cs.add_hyperparameter(hidden_dim_2)

        hidden_dim_3 = CSH.UniformIntegerHyperparameter('hidden_dim_3', lower=100, upper=1000, log=False)
        cs.add_hyperparameter(hidden_dim_3)

        # Use inequality conditions
        cond = CS.GreaterThanCondition(hidden_dim_2, num_hidden_layers, 1)
        cs.add_condition(cond)

        cond = CS.GreaterThanCondition(hidden_dim_3, num_hidden_layers, 2)
        cs.add_condition(cond)

        return cs


class NeuralNet(torch.nn.Module):
    """
    Just a simple pytorch implementation of a feed forward network.
    """
    def __init__(self, input_dim, num_hidden_layers, hidden_dim_1, hidden_dim_2, hidden_dim_3, output_dim, act_f):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = None
        self.fc3 = None

        if num_hidden_layers >= 2:
            self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        if num_hidden_layers == 3:
            self.fc3 = torch.nn.Linear(hidden_dim_2, hidden_dim_3)

        last_hidden_dim = hidden_dim_1 if num_hidden_layers == 1 else \
            hidden_dim_2 if num_hidden_layers == 2 else hidden_dim_3

        self.fc_out = torch.nn.Linear(last_hidden_dim, output_dim)

        if act_f == 'ReLU':
            self.act_f = torch.nn.ReLU()
        elif act_f == 'Tanh':
            self.act_f = torch.nn.Tanh()

    def forward(self, x):
        x = self.act_f(self.fc1(x))
        if self.fc2 is not None:
            x = self.act_f(self.fc2(x))
        if self.fc3 is not None:
            x = self.act_f(self.fc3(x))
        x = self.fc_out(x)
        return x


if __name__ == "__main__":
    worker = MyWorker(run_id='0')
    cs = MyWorker.get_configspace()
    print(cs.sample_configuration())
