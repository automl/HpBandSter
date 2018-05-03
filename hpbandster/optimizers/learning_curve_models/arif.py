import sys
sys.path.append("../../")

import numpy as np
from sklearn.ensemble import RandomForestRegressor as rfr

from hpbandster.learning_curve_models.base import LCModel as lcm_base

from IPython import embed

class ARIF(lcm_base):
    """
        An 'Auto Regressive Integrated (Random) Forest'
    """
    def __init__(self, order=2, diff_order=0):
        """
            Parameters:
            -----------

            order: int
                the order of the 'autoregressiveness'
            diff_order: int
                the differencing order used
                TODO: Not used so far!
        """
        self.order = order
        self.diff_order = diff_order

    def apply_differencing(self, series, order=None):

        if order is None: order = self.diff_order

        for o in range(order):
            series = series[1:]-series[:-1] 
        return series
            

    def invert_differencing(self, initial_part, differenced_rest, order=None):
        """
            function to invert the differencing
        """

        if order is None: order = self.diff_order

        # compute the differenced values of the initial part:
        starting_points = [ self.apply_differencing(initial_part, order=order)[-1] for order in range(self.diff_order)]
        
        actual_predictions = differenced_rest
        import pdb
        pdb.set_trace()
        for s in starting_points[::-1]:
            actual_predictions = np.cumsum(np.hstack([s, actual_predictions]))[1:]

        return(actual_predictions)

    def fit(self, losses, configs=None):

        if configs is None:
            configs = [[]]*len(times)

        # convert learning curves into X and y data

        X = []
        y = []

        for l,c in zip(losses, configs):
            l = self.apply_differencing(l)

            for i in range(self.order, len(l)):
                X.append(np.hstack([l[i-self.order:i], c]))
                y.append(l[i])

        self.X = np.array(X)
        self.y = np.array(y)


        self.rfr = rfr().fit(self.X,self.y)

        
    def extend_partial(self, obs_losses, num_steps, config=None):
        # TODO: add variance predictions
        if config is None:
            config = []

        d_losses = self.apply_differencing(obs_losses)


        for t in range(num_steps):
            x = np.hstack([d_losses[-self.order:], config])
            y = self.rfr.predict([x])
            d_losses = np.hstack([d_losses, y])


        prediction = self.invert_differencing( obs_losses, d_losses[-num_steps:])

        return(prediction)


if __name__ == "__main__":

    sys.path.append("/home/sfalkner/repositories/bitbucket/learning_curve_prediction")
    

    
    from lc_prediction.utils import load_configs
    
    
    #data = load_configs("/home/sfalkner/repositories/bitbucket/learning_curve_prediction/data/conv_net_cifar10", 256+128)
    data = load_configs("/home/sfalkner/repositories/bitbucket/learning_curve_prediction/data/fc_net_mnist", 1024)
    #data = load_configs("/home/sfalkner/repositories/bitbucket/learning_curve_prediction/data/lr_mnist", 1024)
    
    data = (data[0], data[1][:,:40])
    
    import matplotlib.pyplot as plt
    
    #plt.plot(data[1].T)
    #plt.show()
    
    
    full_lcs =  [ lc for lc in data[1]]
    
    T_max = len(full_lcs[0])
    
    learning_curves = [ lc[:np.random.randint(lc.shape[0]-8) + 8]for lc in data[1]]
    #learning_curves = [ lc[:4+ int(np.random.exponential(5))] for lc in data[1]]
    times = [np.arange(1, lc.shape[0]+1) for lc in learning_curves]
    
    lc_model = ARIF(order=3, diff_order=2)
    
    
    
    test_order = 2
    random_sequence = np.random.rand(5)
    tmp = lc_model.apply_differencing(random_sequence, order=test_order)
    
    for i in range(test_order+1):
        print(lc_model.apply_differencing(random_sequence, order=i))
    reconstruction = lc_model.invert_differencing(random_sequence[:1+test_order], tmp, order=test_order)
    
    embed()
    
    
    
    lc_model.fit(learning_curves, data[0])
    
    for i in range(16):
        pred_times = range(times[i][-1]+1, T_max)
        #pred = lc_model.extend_partial(learning_curves[i], min(10, T_max - len(learning_curves[i])), config=data[0][i])
        pred = lc_model.extend_partial(learning_curves[i], T_max - len(learning_curves[i]), config=data[0][i])
        plt.plot(full_lcs[i])
        plt.plot(range(len(learning_curves[i]), len(learning_curves[i])+ len(pred)), pred, '--')
    plt.show()
    
    embed()
    
    
    
