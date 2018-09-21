from sklearn import datasets, neural_network, metrics
from hpbandster.core.worker import Worker

# inherit from the hpbandster.core.worker class
class MyWorker(Worker):
    """ This is a worker for the jupyter-notebook that shows how to connect BOHB to CAVE. """
    def __init__(self, *args, **kwargs):
        super(MyWorker, self).__init__(*args, **kwargs)
        digits = datasets.load_digits()  # load the digits dataset
        n_samples = len(digits.images)
        data = digits.images.reshape((n_samples, -1))

        # split it into training and validation set.
        split = n_samples // 2
        self.train_x, self.valid_x = data[:split], data[split:]
        self.train_y, self.valid_y = digits.target[:split], digits.target[split:]

    def compute(self, config, budget, *args, **kwargs):
        """ overwrite the *compute* methode: the training of the model happens here """
        beta_1 = 0  if 'beta_1' not in config else config['beta_1']
        beta_2 = 0  if 'beta_2' not in config else config['beta_2']

        clf = neural_network.MLPClassifier(max_iter=int(budget),
                                           learning_rate='constant',
                                           learning_rate_init=config['learning_rate_init'],
                                           activation=config['activation'],
                                           solver=config['solver'],
                                           beta_1=beta_1,
                                           beta_2=beta_2
                                          )
        clf.fit(self.train_x, self.train_y)

        predicted = clf.predict(self.valid_x)
        loss_train = metrics.log_loss(self.train_y, clf.predict_proba(self.train_x))
        loss_valid = metrics.log_loss(self.valid_y, clf.predict_proba(self.valid_x))

        accuracy_train = clf.score(self.train_x, self.train_y)
        accuracy_valid = clf.score(self.valid_x, self.valid_y)

        # make sure that the returned dictionary contains the fields *loss* and *info*
        return ({
            'loss': loss_valid,  # this is the a mandatory field to run hyperband
            'info': {'loss_train': loss_train,
                     'loss_test': loss_valid,
                     'accuracy_train': accuracy_train,
                     'accuracy_test': accuracy_valid,
                    }  # can be used for any user-defined information - also mandatory
        })


import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

def get_configspace():
    """ Returns the configuration space for the network to be configured in the example. """
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(CSH.CategoricalHyperparameter('activation', ['tanh', 'relu']))
    config_space.add_hyperparameter(CS.UniformFloatHyperparameter('learning_rate_init', lower=1e-6, upper=1e-2, log=True))
    
    solver = CSH.CategoricalHyperparameter('solver', ['sgd', 'adam'])
    config_space.add_hyperparameter(solver)
    
    beta_1 = CS.UniformFloatHyperparameter('beta_1', lower=0, upper=1)
    config_space.add_hyperparameter(beta_1)
    
    condition = CS.EqualsCondition(beta_1, solver, 'adam')
    config_space.add_condition(condition)
    
    beta_2 = CS.UniformFloatHyperparameter('beta_2', lower=0, upper=1)
    config_space.add_hyperparameter(beta_2)
    
    condition = CS.EqualsCondition(beta_2, solver, 'adam')
    config_space.add_condition(condition)
    
    return config_space
