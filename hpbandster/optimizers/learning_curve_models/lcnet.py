import numpy as np

from hpbandster.learning_curve_models.base import LCModel

from robo.models.lcnet import LCNet, get_lc_net


class LCNetWrapper(LCModel):
    """
        Wrapper around LC-Net
    """

    def __init__(self, max_num_epochs):
        self.max_num_epochs = max_num_epochs
        self.model = LCNet(sampling_method="sghmc",
                           l_rate=np.sqrt(1e-4),
                           mdecay=.05,
                           n_nets=100,
                           burn_in=5000,
                           n_iters=30000,
                           get_net=get_lc_net,
                           precondition=True)
    
    def fit(self, times, losses, configs=None):
        """
            function to train the model on the observed data

            Parameters:
            -----------

            times: list
                list of numpy arrays of the timesteps for each curve
            losses: list
                list of numpy arrays of the loss (the actual learning curve)
            configs: list or None
                list of the configurations for each sample. Each element
                has to be a numpy array. Set to None, if no configuration
                information is available.
        """

        assert np.all(times > 0) and np.all(times <= self.max_num_epochs)

        train = None
        targets = None

        for i in range(len(configs)):

            t_idx = times[i] / self.max_num_epochs

            x = np.repeat(np.array(configs[i])[None, :], t_idx.shape[0], axis=0)
            x = np.concatenate((x, t_idx[:, None]), axis=1)

            # LCNet assumes increasing curves, if we feed in losses here we have to flip the curves
            lc = [1 - l for l in losses[i]]

            if train is None:
                train = x
                targets = lc
            else:
                train = np.concatenate((train, x), 0)
                targets = np.concatenate((targets, lc), 0)

        self.model.train(train, targets)

    def predict_unseen(self, times, config):
        """
            predict the loss of an unseen configuration

            Parameters:
            -----------

            times: numpy array
                times where to predict the loss
            config: numpy array
                the numerical representation of the config

            Returns:
            --------
            
            mean and variance prediction at input times for the given config
        """

        assert np.all(times > 0) and np.all(times <= self.max_num_epochs)

        x = np.array(config)[None, :]

        idx = times / self.max_num_epochs
        x = np.repeat(x, idx.shape[0], axis=0)

        x = np.concatenate((x, idx[:, None]), axis=1)

        mean, var = self.model.predict(x)
        return 1 - mean, var

    def extend_partial(self, times, obs_times, obs_losses, config=None):
        """
            extends a partially observed curve

            Parameters:
            -----------

            times: numpy array
                times where to predict the loss
            obs_times: numpy array
                times where the curve has already been observed
            obs_losses: numpy array
                corresponding observed losses
            config: numpy array
                numerical reperesentation of the config; None if no config
                information is available
                
            Returns:
            --------
            
            mean and variance prediction at input times
                
                
        """
        return self.predict_unseen(times, config)

