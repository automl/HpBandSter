import ConfigSpace
import numpy as np
import threading

from robo.models.lcnet import LCNet, get_lc_net

from hpbandster.core.base_config_generator import base_config_generator


def smoothing(lc):
    new_lc = []
    curr_best = np.inf
    for i in range(len(lc)):
        if lc[i] < curr_best:
            curr_best = lc[i]
        new_lc.append(curr_best)
    return new_lc


class LCNetWrapper(base_config_generator):
    def __init__(self,
                 configspace,
                 max_budget,
                 n_points=2000,
                 delta=1.0,
                 n_candidates=1024,
                 **kwargs):
        """
        Parameters:
        -----------

        directory: string
            where the results are logged
        logger: hpbandster.utils.result_logger_v??
            the logger to store the data, defaults to v1
        overwrite: bool
            whether or not existing data will be overwritten

        """

        super(LCNetWrapper, self).__init__(**kwargs)

        self.n_candidates = n_candidates
        self.model = LCNet(sampling_method="sghmc",
                           l_rate=np.sqrt(1e-4),
                           mdecay=.05,
                           n_nets=100,
                           burn_in=500,
                           n_iters=3000,
                           get_net=get_lc_net,
                           precondition=True)

        self.config_space = configspace
        self.max_budget = max_budget
        self.train = None
        self.train_targets = None
        self.n_points = n_points
        self.is_trained = False
        self.counter = 0
        self.delta = delta
        self.lock = threading.Lock()

    def get_config(self, budget):
        """
            function to sample a new configuration

            This function is called inside Hyperband to query a new configuration


            Parameters:
            -----------
            budget: float
                the budget for which this configuration is scheduled

            returns: config
                should return a valid configuration

        """
        self.lock.acquire()
        if not self.is_trained:
            c = self.config_space.sample_configuration().get_array()
        else:
            candidates = np.array([self.config_space.sample_configuration().get_array()
                                   for _ in range(self.n_candidates)])

            # We are only interested on the asymptotic value
            projected_candidates = np.concatenate((candidates, np.ones([self.n_candidates, 1])), axis=1)

            # Compute the upper confidence bound of the function at the asymptote
            m, v = self.model.predict(projected_candidates)

            ucb_values = m + self.delta * np.sqrt(v)
            print(ucb_values)
            # Sample a configuration based on the ucb values
            p = np.ones(self.n_candidates) * (ucb_values / np.sum(ucb_values))
            idx = np.random.choice(self.n_candidates, 1, False, p)

            c = candidates[idx][0]

        config = ConfigSpace.Configuration(self.config_space, vector=c)

        self.lock.release()
        return config.get_dictionary(), {}

    def new_result(self, job):
        """
            function to register finished runs

            Every time a run has finished, this function should be called
            to register it with the result logger. If overwritten, make
            sure to call this method from the base class to ensure proper
            logging.


            Parameters:
            -----------
            job_id: dict
                a dictionary containing all the info about the run
            job_result: dict
                contains all the results of the job, i.e. it's a dict with
                the keys 'loss' and 'info'

        """
        super().new_result(job)

        conf = ConfigSpace.Configuration(self.config_space, job.kwargs['config']).get_array()

        epochs = len(job.result["info"]["learning_curve"])
        budget = int(job.kwargs["budget"])

        t_idx = np.linspace(budget / epochs, budget, epochs) / self.max_budget
        x_new = np.repeat(conf[None, :], t_idx.shape[0], axis=0)

        x_new = np.concatenate((x_new, t_idx[:, None]), axis=1)

        # Smooth learning curve
        lc = smoothing(job.result["info"]["learning_curve"])

        # Flip learning curves since LC-Net wants increasing curves
        lc_new = [1 - y for y in lc]

        if self.train is None:
            self.train = x_new
            self.train_targets = lc_new
        else:
            self.train = np.append(self.train, x_new, axis=0)
            self.train_targets = np.append(self.train_targets, lc_new, axis=0)

        if self.counter >= self.n_points:

            self.lock.acquire()
            y_min = np.min(self.train_targets)
            y_max = np.max(self.train_targets)

            train_targets = (self.train_targets - y_min) / (y_max - y_min)

            self.model.train(self.train, train_targets)
            self.is_trained = True
            self.counter = 0
            self.lock.release()

        else:
            self.counter += epochs
