# A more involved example of training a RNN on the 20 Newsgroups dataset.
# The results are not really good, but a RNN on character level and with
# all the the simple features ('header', 'footer' and 'quotes') removed,
# this is actually a hard problem, especially if no word embeddings are
# used.
# The purpose of this example is to show how a more complicated worker
# could look like.

# NOTE: obviously, the implementation is very inefficient as every document
# gets converted into a tensor on the fly. For better performance, this
# should probably be done once and then reused


import time
import os, sys

import string

import logging
logging.basicConfig(level=logging.DEBUG)

from hpbandster.core.worker import Worker

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score


import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


class GRU_classifier(nn.Module):
    """
    This is a pytorch implementation of a RNN for the 20 news dataset.
    """
    def __init__(self, input_size, gru_num_layers, gru_size, mlp_size1, mlp_size2, num_classes):
        super(GRU_classifier, self).__init__()

        self.input_size = input_size
        self.gru_num_layers = gru_num_layers
        self.gru_size = gru_size
        self.mlp_size1 = mlp_size1
        self.mlp_size2 = mlp_size2
        self.output_size = num_classes

        self.gru = nn.GRU(input_size, gru_size, num_layers = gru_num_layers)
        self.hidden1 = nn.Linear(gru_size, mlp_size1)
        self.hidden2 = nn.Linear(mlp_size1, mlp_size2)
        self.output = nn.Linear(mlp_size2, num_classes)

    def init_hidden(self):
        self.hidden_state = Variable(torch.zeros(self.gru_num_layers, 1, self.gru_size))

    def forward(self, input):
        gru_out, self.hidden_state = self.gru(input.view(len(input), 1, -1), self.hidden_state)
        hidden_out1 = self.hidden1(gru_out.view(len(input),-1)[-1:])

        hidden_out2 = self.hidden2(F.sigmoid(hidden_out1))
        return self.output(hidden_out2)


class RNN20NGWorker(Worker):
    """
    Inherited class of hpbandster.core.worker.
    This class implements the connection between pytorch-net and the optimizer.
    """
    def __init__(self, cutoff = 128,                # documents will be cut to that length
                 categories=[                            # only use a subset of categories
                 #'alt.atheism',
                 'talk.religion.misc',
                 #'comp.graphics',
                 #'sci.space',
                 'rec.sport.baseball',
                 #'rec.autos',
                 #'rec.motorcycles',
                 ], *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.cutoff = cutoff
        self.categories = categories

        self.test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=self.categories)
        self.train_data= fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=self.categories)

        self.n_classes = len(self.test_data.target_names)

        all_letters = string.printable
        all_letters = sorted(list(set(all_letters.lower())))
        self.all_letters = ''.join(all_letters)
        self.n_letters = len(all_letters)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
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

        current_loss = 0
        all_losses = []

        average_interval = 64

        self.rnn = GRU_classifier(len(self.all_letters),
                                  config['gru_layers'], config['gru_size'],
                                  config['hddn1'], config['hddn2'], self.n_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.rnn.parameters(), lr=config['lr'], momentum=config['momentum'])

        for epoch in range(int(budget)*average_interval):
            line = ''
            while len(line) == 0:
                category, line, category_tensor, line_tensor = self._random_training_pair(self.cutoff)

            self.rnn.zero_grad()
            self.rnn.init_hidden()

            output = self.rnn(line_tensor)

            loss = criterion(output, category_tensor)
            loss.backward()
            optimizer.step()

            current_loss += loss.data.item()

            if epoch % average_interval == 0:
                all_losses.append(current_loss / average_interval)
                self.logger.debug('epoch %i, current loss: %f'%(epoch, all_losses[-1]))
                current_loss = 0

        print('done training')
        train_acc = self._compute_train_accuracy()
        print('done computing train_accuracy')
        test_acc = self._compute_test_accuracy()

        return({'loss': -test_acc,
                'info': {
                       'train_acc': train_acc,
                       'test_acc' : test_acc,
                       'learning_curve': all_losses,
                        }
               })

    @staticmethod
    def get_config_space():
        """
        Here we define the configuration space for the hyperparameters for the model.
        For a more complex example please see the configuration space - example in the documentation
        Returns:
            ConfigSpace-object
        """
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('gru_layers', lower=1, upper=3, log=True))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('gru_size', lower=16, upper=128, log=True))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('hddn1', lower=32, upper=512, log=True))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter('hddn2', lower=32, upper=512, log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('lr', lower=1e-8, upper=1e-6, log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter('momentum', lower=1e-5, upper=0.99,))
        return cs

    ####################################################################
    # helper function for pytorch RNN below this line
    def _line_to_tensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            letter_index = self.all_letters.find(letter)
            tensor[li][0][letter_index] = 1
        return tensor

    def _random_training_pair(self, cut_length = 1024, index=None):
        if index is None:
            index = random.randint(0,len(self.train_data.data)-1)

        doc = self.train_data.data[index][:cut_length].lower()

        category_tensor = Variable(torch.LongTensor([int(self.train_data.target[index])]))
        line_tensor = Variable(self._line_to_tensor(doc))
        return self.train_data.target[index], doc, category_tensor, line_tensor


    def _random_test_pair(self, cut_length = 1024, index=None):
        if index is None:
            index = random.randint(0,len(self.test_data.data)-1)
        doc = self.train_data.data[index][:cut_length].lower()

        category_tensor = Variable(torch.LongTensor([int(self.test_data.target[index])]))
        line_tensor = Variable(self._line_to_tensor(doc))
        return self.train_data.target[index], doc, category_tensor, line_tensor


    def _compute_train_accuracy(self):
        predictions = []
        truths = []

        self.rnn.eval()

        for index in range(len(self.train_data.data)):
            category, line, category_tensor, line_tensor = self._random_training_pair(self.cutoff, index=index)
            if len(line) == 0:
                    continue
            self.rnn.init_hidden()
            p = self.rnn(line_tensor)
            predictions.append(p.data.topk(1)[1][0][0])
            truths.append(self.train_data.target[index])
        truths = np.array(truths)
        predictions = np.array(predictions)
        return(accuracy_score(predictions, truths))

    def _compute_test_accuracy(self):
        predictions = []
        truths = []

        self.rnn.eval()

        for index in range(len(self.test_data.data)):
            category, line, category_tensor, line_tensor = self._random_test_pair(self.cutoff, index=index)
            if len(line) == 0:
                    continue
            self.rnn.init_hidden()
            p = self.rnn(line_tensor)
            predictions.append(p.data.topk(1)[1][0][0])
            truths.append(self.test_data.target[index])
        truths = np.array(truths)
        predictions = np.array(predictions)

        return accuracy_score(predictions, truths)


if __name__ == "__main__":
    worker = RNN20NGWorker(run_id='0')
    cs = RNN20NGWorker.get_config_space()

    config = cs.sample_configuration()
    print(config)
    print(worker.compute(config.get_dictionary(), 16, '.'))
