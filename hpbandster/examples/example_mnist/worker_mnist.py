import torch
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


#
# In this example we'll show:
#     1) how to connect a worker with a neural network.
#     2) implement a more complex ConfigSpace with conditional hyperparameters
#
# We'll optimise the following hyperparameters:
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
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        # device = torch.device('cpu')

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

            for i, (x, y) in enumerate(self.train_loader):
                optimizer.zero_grad()

                x = x.reshape(-1, 28*28)

                output = model(x)
                err = criterion(output, y)
                loss += err.data.item()

                err.backward()
                optimizer.step()

            print('Epoch [{:4}|{:4}] Loss: {:10.4f}'
                  .format(epoch+1, int(budget), loss))

        return ({
            'loss': loss,  # this is the a mandatory field to run hyperband
            'info': {'loss': 'value you like to store'}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
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

        num_hidden_layers = CSH.UniformIntegerHyperparameter('num_hidden_layers', lower=1, upper=3, default_value=1)
        cs.add_hyperparameter(num_hidden_layers)

        hidden_dim_1 = CSH.UniformIntegerHyperparameter('hidden_dim_1', lower=100, upper=1000, log=False)
        cs.add_hyperparameter(hidden_dim_1)

        hidden_dim_2 = CSH.UniformIntegerHyperparameter('hidden_dim_2', lower=100, upper=1000, log=False)
        cs.add_hyperparameter(hidden_dim_2)

        hidden_dim_3 = CSH.UniformIntegerHyperparameter('hidden_dim_3', lower=100, upper=1000, log=False)
        cs.add_hyperparameter(hidden_dim_3)



        # You can also use inequality conditions:
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